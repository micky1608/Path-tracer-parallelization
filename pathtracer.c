/* basé sur on smallpt, a Path Tracer by Kevin Beason, 2008
 *  	http://www.kevinbeason.com/smallpt/ 
 *
 * Converti en C et modifié par Charles Bouillaguet, 2019
 *
 * Pour des détails sur le processus de rendu, lire :
 * 	https://docs.google.com/open?id=0B8g97JkuSSBwUENiWTJXeGtTOHFmSm51UC01YWtCZw
 */

#define _XOPEN_SOURCE
#include <mpi.h>
#include <math.h>   
#include <stdlib.h> 
#include <stdio.h>
#include <stdbool.h>
#include <sys/time.h>
#include <sys/stat.h>  /* pour mkdir    */ 
#include <unistd.h>    /* pour getuid   */
#include <sys/types.h> /* pour getpwuid */
#include <pwd.h>       /* pour getpwuid */
#include <time.h>

#define SIZE_BLOCK 32	// nombre de pixels contenus dans un bloc
#define SIZE_PIXEL 3*sizeof(double)

//TAGS
#define TAG_NEED_DATA 10
#define TAG_YEP 11
#define TAG_NOPE 12
#define TAG_TASK_INFO 13
#define TAG_TAST_DATA 14
#define TAG_DATA 42

enum Refl_t {DIFF, SPEC, REFR};   /* types de matériaux (DIFFuse, SPECular, REFRactive) */

struct Sphere { 
	double radius; 
	double position[3];
	double emission[3];     /* couleur émise (=source de lumière) */
	double color[3];        /* couleur de l'objet RGB (diffusion, refraction, ...) */
	enum Refl_t refl;       /* type de reflection */
	double max_reflexivity;
};

static const int KILL_DEPTH = 7;
static const int SPLIT_DEPTH = 4;

/* la scène est composée uniquement de spheres */
struct Sphere spheres[] = { 
// radius position,                         emission,     color,              material 
   {1e5,  { 1e5+1,  40.8,       81.6},      {},           {.75,  .25,  .25},  DIFF, -1}, // Left 
   {1e5,  {-1e5+99, 40.8,       81.6},      {},           {.25,  .25,  .75},  DIFF, -1}, // Right 
   {1e5,  {50,      40.8,       1e5},       {},           {.75,  .75,  .75},  DIFF, -1}, // Back 
   {1e5,  {50,      40.8,      -1e5 + 170}, {},           {},                 DIFF, -1}, // Front 
   {1e5,  {50,      1e5,        81.6},      {},           {0.75, .75,  .75},  DIFF, -1}, // Bottom 
   {1e5,  {50,     -1e5 + 81.6, 81.6},      {},           {0.75, .75,  .75},  DIFF, -1}, // Top 
   {16.5, {40,      16.5,       47},        {},           {.999, .999, .999}, SPEC, -1}, // Mirror 
   {16.5, {73,      46.5,       88},        {},           {.999, .999, .999}, REFR, -1}, // Glass 
   {10,   {15,      45,         112},       {},           {.999, .999, .999}, DIFF, -1}, // white ball
   {15,   {16,      16,         130},       {},           {.999, .999, 0},    REFR, -1}, // big yellow glass
   {7.5,  {40,      8,          120},        {},           {.999, .999, 0   }, REFR, -1}, // small yellow glass middle
   {8.5,  {60,      9,          110},        {},           {.999, .999, 0   }, REFR, -1}, // small yellow glass right
   {10,   {80,      12,         92},        {},           {0, .999, 0},       DIFF, -1}, // green ball
   {600,  {50,      681.33,     81.6},      {12, 12, 12}, {},                 DIFF, -1},  // Light 
   {5,    {50,      75,         81.6},      {},           {0, .682, .999}, DIFF, -1}, // occlusion, mirror
}; 


/********** micro BLAS LEVEL-1 + quelques fonctions non-standard **************/
static inline void copy(const double *x, double *y)
{
	for (int i = 0; i < 3; i++)
		y[i] = x[i];
} 

static inline void zero(double *x)
{
	for (int i = 0; i < 3; i++)
		x[i] = 0;
} 

static inline void axpy(double alpha, const double *x, double *y)
{
	for (int i = 0; i < 3; i++)
		y[i] += alpha * x[i];
} 

static inline void scal(double alpha, double *x)
{
	for (int i = 0; i < 3; i++)
		x[i] *= alpha;
} 

static inline double dot(const double *a, const double *b)
{ 
	return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
} 

static inline double nrm2(const double *a)
{
	return sqrt(dot(a, a));
}

/********* fonction non-standard *************/
static inline void mul(const double *x, const double *y, double *z)
{
	for (int i = 0; i < 3; i++)
		z[i] = x[i] * y[i];
} 

static inline void normalize(double *x)
{
	scal(1 / nrm2(x), x);
}

/* produit vectoriel */
static inline void cross(const double *a, const double *b, double *c)
{
	c[0] = a[1] * b[2] - a[2] * b[1];
	c[1] = a[2] * b[0] - a[0] * b[2];
	c[2] = a[0] * b[1] - a[1] * b[0];
}

/****** tronque *************/
static inline void clamp(double *x)
{
	for (int i = 0; i < 3; i++) {
		if (x[i] < 0)
			x[i] = 0;
		if (x[i] > 1)
			x[i] = 1;
	}
} 

/******************************* calcul des intersections rayon / sphere *************************************/
   
// returns distance, 0 if nohit 
double sphere_intersect(const struct Sphere *s, const double *ray_origin, const double *ray_direction)
{ 
	double op[3];
	// Solve t^2*d.d + 2*t*(o-p).d + (o-p).(o-p)-R^2 = 0 
	copy(s->position, op);
	axpy(-1, ray_origin, op);
	double eps = 1e-4;
	double b = dot(op, ray_direction);
	double discriminant = b * b - dot(op, op) + s->radius * s->radius; 
	if (discriminant < 0)
		return 0;   /* pas d'intersection */
	else 
		discriminant = sqrt(discriminant);
	/* détermine la plus petite solution positive (i.e. point d'intersection le plus proche, mais devant nous) */
	double t = b - discriminant;
	if (t > eps) {
		return t;
	} else {
		t = b + discriminant;
		if (t > eps)
			return t;
		else
			return 0;  /* cas bizarre, racine double, etc. */
	}
}

/* détermine si le rayon intersecte l'une des spere; si oui renvoie true et fixe t, id */
bool intersect(const double *ray_origin, const double *ray_direction, double *t, int *id)
{ 
	int n = sizeof(spheres) / sizeof(struct Sphere);
	double inf = 1e20; 
	*t = inf;
	for (int i = 0; i < n; i++) {
		double d = sphere_intersect(&spheres[i], ray_origin, ray_direction);
		if ((d > 0) && (d < *t)) {
			*t = d;
			*id = i;
		} 
	}
	return *t < inf;
} 

/* calcule (dans out) la luminance reçue par la camera sur le rayon donné */
void radiance(const double *ray_origin, const double *ray_direction, int depth, unsigned short *PRNG_state, double *out)
{ 
	int id = 0;                             // id de la sphère intersectée par le rayon
	double t;                               // distance à l'intersection
	if (!intersect(ray_origin, ray_direction, &t, &id)) {
		zero(out);    // if miss, return black 
		return; 
	}
	const struct Sphere *obj = &spheres[id];
	
	/* point d'intersection du rayon et de la sphère */
	double x[3];
	copy(ray_origin, x);
	axpy(t, ray_direction, x);
	
	/* vecteur normal à la sphere, au point d'intersection */
	double n[3];  
	copy(x, n);
	axpy(-1, obj->position, n);
	normalize(n);
	
	/* vecteur normal, orienté dans le sens opposé au rayon 
	   (vers l'extérieur si le rayon entre, vers l'intérieur s'il sort) */
	double nl[3];
	copy(n, nl);
	if (dot(n, ray_direction) > 0)
		scal(-1, nl);
	
	/* couleur de la sphere */
	double f[3];
	copy(obj->color, f);
	double p = obj->max_reflexivity;

	/* processus aléatoire : au-delà d'une certaine profondeur,
	   décide aléatoirement d'arrêter la récusion. Plus l'objet est
	   clair, plus le processus a de chance de continuer. */
	depth++;
	if (depth > KILL_DEPTH) {
		if (erand48(PRNG_state) < p) {
			scal(1 / p, f); 
		} else {
			copy(obj->emission, out);
			return;
		}
	}

	/* Cas de la réflection DIFFuse (= non-brillante). 
	   On récupère la luminance en provenance de l'ensemble de l'univers. 
	   Pour cela : (processus de monte-carlo) on choisit une direction
	   aléatoire dans un certain cone, et on récupère la luminance en 
	   provenance de cette direction. */
	if (obj->refl == DIFF) {
		double r1 = 2 * M_PI * erand48(PRNG_state);  /* angle aléatoire */
		double r2 = erand48(PRNG_state);             /* distance au centre aléatoire */
		double r2s = sqrt(r2); 
		
		double w[3];   /* vecteur normal */
		copy(nl, w);
		
		double u[3];   /* u est orthogonal à w */
		double uw[3] = {0, 0, 0};
		if (fabs(w[0]) > .1)
			uw[1] = 1;
		else
			uw[0] = 1;
		cross(uw, w, u);
		normalize(u);
		
		double v[3];   /* v est orthogonal à u et w */
		cross(w, u, v);
		
		double d[3];   /* d est le vecteur incident aléatoire, selon la bonne distribution */
		zero(d);
		axpy(cos(r1) * r2s, u, d);
		axpy(sin(r1) * r2s, v, d);
		axpy(sqrt(1 - r2), w, d);
		normalize(d);
		
		/* calcule récursivement la luminance du rayon incident */
		double rec[3];
		radiance(x, d, depth, PRNG_state, rec);
		
		/* pondère par la couleur de la sphère, prend en compte l'emissivité */
		mul(f, rec, out);
		axpy(1, obj->emission, out);
		return;
	}

	/* dans les deux autres cas (réflection parfaite / refraction), on considère le rayon
	   réfléchi par la spère */

	double reflected_dir[3];
	copy(ray_direction, reflected_dir);
	axpy(-2 * dot(n, ray_direction), n, reflected_dir);

	/* cas de la reflection SPEculaire parfaire (==mirroir) */
	if (obj->refl == SPEC) { 
		double rec[3];
		/* calcule récursivement la luminance du rayon réflechi */
		radiance(x, reflected_dir, depth, PRNG_state, rec);
		/* pondère par la couleur de la sphère, prend en compte l'emissivité */
		mul(f, rec, out);
		axpy(1, obj->emission, out);
		return;
	}

	/* cas des surfaces diélectriques (==verre). Combinaison de réflection et de réfraction. */
	bool into = dot(n, nl) > 0;      /* vient-il de l'extérieur ? */
	double nc = 1;                   /* indice de réfraction de l'air */
	double nt = 1.5;                 /* indice de réfraction du verre */
	double nnt = into ? (nc / nt) : (nt / nc);
	double ddn = dot(ray_direction, nl);
	
	/* si le rayon essaye de sortir de l'objet en verre avec un angle incident trop faible,
	   il rebondit entièrement */
	double cos2t = 1 - nnt * nnt * (1 - ddn * ddn);
	if (cos2t < 0) {
		double rec[3];
		/* calcule seulement le rayon réfléchi */
		radiance(x, reflected_dir, depth, PRNG_state, rec);
		mul(f, rec, out);
		axpy(1, obj->emission, out);
		return;
	}
	
	/* calcule la direction du rayon réfracté */
	double tdir[3];
	zero(tdir);
	axpy(nnt, ray_direction, tdir);
	axpy(-(into ? 1 : -1) * (ddn * nnt + sqrt(cos2t)), n, tdir);

	/* calcul de la réflectance (==fraction de la lumière réfléchie) */
	double a = nt - nc;
	double b = nt + nc;
	double R0 = a * a / (b * b);
	double c = 1 - (into ? -ddn : dot(tdir, n));
	double Re = R0 + (1 - R0) * c * c * c * c * c;   /* réflectance */
	double Tr = 1 - Re;                              /* transmittance */
	
	/* au-dela d'une certaine profondeur, on choisit aléatoirement si
	   on calcule le rayon réfléchi ou bien le rayon réfracté. En dessous du
	   seuil, on calcule les deux. */
	double rec[3];
	if (depth > SPLIT_DEPTH) {
		double P = .25 + .5 * Re;             /* probabilité de réflection */
		if (erand48(PRNG_state) < P) {
			radiance(x, reflected_dir, depth, PRNG_state, rec);
			double RP = Re / P;
			scal(RP, rec);
		} else {
			radiance(x, tdir, depth, PRNG_state, rec);
			double TP = Tr / (1 - P); 
			scal(TP, rec);
		}
	} else {
		double rec_re[3], rec_tr[3];
		radiance(x, reflected_dir, depth, PRNG_state, rec_re);
		radiance(x, tdir, depth, PRNG_state, rec_tr);
		zero(rec);
		axpy(Re, rec_re, rec);
		axpy(Tr, rec_tr, rec);
	}
	/* pondère, prend en compte la luminance */
	mul(f, rec, out);
	axpy(1, obj->emission, out);
	return;
}

double wtime()
{
	struct timeval ts;
	gettimeofday(&ts, NULL);
	return (double)ts.tv_sec + ts.tv_usec / 1E6;
}

int toInt(double x)
{
	return pow(x, 1 / 2.2) * 255 + .5;   /* gamma correction = 2.2 */
}

//Melange un tableau
void shuffle(int *array, size_t n, int rank)
{
	unsigned short PRNG_state[3] = {0, rank, rank*rank};
    if (n > 1) 
    {
        size_t i;
        for (i = 0; i < n - 1; i++) 
        {
          size_t j = i + erand48(PRNG_state)*(n - i);
          int t = array[j];
          array[j] = array[i];
          array[i] = t;
        }
    }
}

int main(int argc, char **argv)
{ 
	// Mesurer le temps d'execution
	double clock_begin, clock_end;
	clock_begin = wtime();

	/* Petit cas test (small, quick and dirty): */
	int w = 320;
	int h = 200;
	int samples = 50;

	/* Gros cas test (big, slow and pretty): */
	/* int w = 3840; */
	/* int h = 2160; */
	/* int samples = 5000;  */

	if (argc == 2) 
		samples = atoi(argv[1]) / 4;

	static const double CST = 0.5135;  /* ceci défini l'angle de vue */
	double camera_position[3] = {50, 52, 295.6};
	double camera_direction[3] = {0, -0.042612, -1};
	normalize(camera_direction);

	/* incréments pour passer d'un pixel à l'autre */
	double cx[3] = {w * CST / h, 0, 0};    
	double cy[3];
	cross(cx, camera_direction, cy);  /* cy est orthogonal à cx ET à la direction dans laquelle regarde la caméra */
	normalize(cy);
	scal(CST, cy);

	/* précalcule la norme infinie des couleurs */
	int n = sizeof(spheres) / sizeof(struct Sphere);
	for (int i = 0; i < n; i++) {
		double *f = spheres[i].color;
		if ((f[0] > f[1]) && (f[0] > f[2]))
			spheres[i].max_reflexivity = f[0]; 
		else {
			if (f[1] > f[2])
				spheres[i].max_reflexivity = f[1];
			else
				spheres[i].max_reflexivity = f[2]; 
		}
	}

	MPI_Init(&argc,&argv);
	MPI_Status status;
	int nbProcess;
	int rank;
	MPI_Comm_size(MPI_COMM_WORLD, &nbProcess);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

		/* Initialisation des variables de division des données*/
	
	// nombre total de blocs 
	int nb_bloc = ceil((double)(w*h) / SIZE_BLOCK);

	// nombre de blocs par processus
	int nb_bloc_local = ceil(nb_bloc / nbProcess);

	// on equilibre les blocs entre les processus
	if(rank < nb_bloc - (nb_bloc_local * nbProcess)) {
		nb_bloc_local++;
	}

	// int h_local = ceil((SIZE_BLOCK*nb_bloc_local)/w);

	if(rank == 0) {
		printf("h : %d pixels\tw : %d pixels\n",h,w);
		printf("nb pixels : %d\n",w*h);
		printf("nb bloc : %d\n",nb_bloc);
	}

	//printf("Process %d : nb_bloc_local = %d\th_local = %d\n",rank , nb_bloc_local,h_local);
	printf("Process %d : nb_bloc_local = %d\n",rank , nb_bloc_local);

	// on stocke les indices des premiers pixels des blocs supplémentaires calculés
	int *pixel_sup;

	// le nombre de blocs supplémentaires calculés
	int nb_sup = 0;

	// tampon mémoire pour l'image
	double *image , *image_sup;

	// tableau pour contenir le nombre de blocs initialement affectés a chaque processus
	int *nb_bloc_locaux;

	// Taille en blocs d'imagesup
	int imageSupSize;

	//Nombre de blocs utilises d'imagesup
	int used_imagesup_blocs=0;


		/* Allocation mémoire */

	if(rank == 0) {
		// allocation de l'image totale
		image = malloc(nb_bloc * SIZE_BLOCK * SIZE_PIXEL);

		// pas de blocs supplementaires
		imageSupSize=0;
		image_sup = NULL;
	}
	else {
		pixel_sup = malloc(nb_bloc_local * sizeof(int));

		if (pixel_sup == NULL) { perror("Impossible d'allouer le tableau des numeros de blocs supplementaires"); exit(1); }

		// allocation d'un tampon principal
		image = malloc(nb_bloc_local * SIZE_BLOCK * SIZE_PIXEL);

		// allocation d'un tampon supplémentaire 
		imageSupSize = nb_bloc_local;
		image_sup = malloc(imageSupSize * SIZE_BLOCK * SIZE_PIXEL);

		if (image_sup == NULL) { perror("Impossible d'allouer le tampon supplémentaire"); exit(1); }
	}

	if (image == NULL) { perror("Impossible d'allouer l'image"); exit(1); }

	nb_bloc_locaux = (int*)malloc(nbProcess*sizeof(int));

	//Rangs des autres processus pour les demandes de données
	int * other_process_ranks = malloc((nbProcess-1)*sizeof(int)); //Tableau des rangs
	char * other_process_hasWork = malloc((nbProcess-1)*sizeof(char)); //Tableau disant si oui ou non ce rang a du travail
	int process_with_work_counter = nbProcess-1; //Compteur des processus ayant du travail
	int other_process_offset=0;
	for(int i=0; i<nbProcess; i++)
	{
		if(rank != i)
		{
			other_process_ranks[other_process_offset] = i;
			other_process_hasWork[other_process_offset]=1;
			other_process_offset++;
		}
	}
	other_process_offset=0;
	shuffle(other_process_ranks, nbProcess-1, rank); //Melange !

	MPI_Allgather(&nb_bloc_local , 1 , MPI_INT , nb_bloc_locaux , 1 , MPI_INT , MPI_COMM_WORLD);

	printf("Process %d : work in progress ...\n",rank);

	// definition des indices dans l'image globale
	int x,y;

	int nb_bloc_precedent = 0;
	for(int l=0 ; l<rank ; l++) nb_bloc_precedent += nb_bloc_locaux[l];

	printf("Process %d : nb_bloc_precedent = %d\n",rank,nb_bloc_precedent);

	//Gestion des taches courantes
	int current_task_start = nb_bloc_precedent;
	int current_task_blocs = nb_bloc_local;
	short processing_local_blocs = 1; //true
	int tasks_offset=0;
	int * tasks = malloc(2*sizeof(int)*nbProcess); // Tableau de taches sous la forme 2 par 2 : nbBlocs, bloc de depart
	int tasks_size = 2*nbProcess;

	//Gestion des requetes
	MPI_Request * recv_requests = malloc(sizeof(MPI_Request)*(nbProcess));
	MPI_Request * send_requests = malloc(sizeof(MPI_Request)*(nbProcess));

	//Gestion des envois de TAG_NOPE
	short * nope_sent = malloc(sizeof(short)*nbProcess);
	for(int i=0; i<nbProcess; i++)
	{
		nope_sent[i] = i==rank;
	}

	int current_sup_offset=0; //Offset pour écrire les pixels dans imagesup
	

	
	/* ****************************************************************************************************************** */
	// Boucle principale 
	/* ****************************************************************************************************************** */

	int dummy; //Variable à envoyer pour les Send
	int task_buffer[2];

	//Lancement des IRecv asynchrones
	for(int i=0; i < nbProcess; i++)
	{
		if(i != rank)
		{
			MPI_Irecv(&dummy, 1, MPI_INT, i, TAG_NEED_DATA, MPI_COMM_WORLD, &recv_requests[i]);
		}
	}

	// Pour chaque bloc
	while(process_with_work_counter > 0)
	{
		//Boucle de demande de taches
		if(!processing_local_blocs)
		{
			while(1)
			{
				if(other_process_hasWork[other_process_offset] == 1) //Si on n'a pas reçu de NOPE de ce processus
				{
					MPI_Send(&dummy, 1, MPI_INT, other_process_ranks[other_process_offset], TAG_NEED_DATA, MPI_COMM_WORLD); //Envoi de la demande
					MPI_Recv(&task_buffer, 2, MPI_INT, other_process_ranks[other_process_offset],MPI_ANY_TAG, MPI_COMM_WORLD, &status); //Reception de la reponse
					if(status.MPI_TAG == TAG_YEP) //On a reçu une tache
					{
						if(tasks_offset == tasks_size) //Si on a plus de place dans la liste des taches
						{
							tasks = realloc(tasks, (tasks_size+2*nbProcess)*sizeof(int)); //On augmente la taille
							tasks_size+=2*nbProcess;
						}
						current_task_blocs = tasks[tasks_offset++] = task_buffer[0];
						current_task_start = tasks[tasks_offset++] = task_buffer[1];
						other_process_offset = (other_process_offset+1)%(nbProcess-1);
						used_imagesup_blocs+=current_task_blocs;
						if(used_imagesup_blocs > imageSupSize) //Si il n'y a plus de place dans le buffer d'image supplementaire
						{
							image_sup = realloc(image_sup, used_imagesup_blocs*SIZE_BLOCK*SIZE_PIXEL);
							imageSupSize=used_imagesup_blocs;
						}
						break;
					}else //On a reçu un NOPE
					{
						other_process_hasWork[other_process_offset] = 0;
						process_with_work_counter--;
						if(process_with_work_counter == 0) //On n'a plus de processus à demander, on stoppe
						{
							break;
						}
					}
				}
				other_process_offset = (other_process_offset+1)%(nbProcess-1);
			}
		}
		if(process_with_work_counter == 0)
		{
			break;
		}
		//for(int b=0; b<nb_bloc_local; b++)
		for(int b=0; b<current_task_blocs; b++)
		{

			if(processing_local_blocs) //Si on traite les blocs locaux
			{
				int nb_blocs_restants = current_task_blocs - b;
				int task = ceil((nb_blocs_restants*1.0)/nbProcess); //Taille des taches
				int finished=0;
				int sendbuffer[2] = {0,0};
				if(nb_blocs_restants > 1) //Si on a des taches à partager
				{
					for(int i=0; i<nbProcess;i++)
					{
						if(i != rank)
						{
							MPI_Test(recv_requests[i], &finished, &status); //On vérifie si on a reçu une demande
							if(finished)
							{
								if(nb_blocs_restants > task) //Si il nous reste au moins une tache à envoyer
								{
									int sendable = task <= nb_blocs_restants-task ? task : nb_blocs_restants-task; //On envoi une tache complete, ou seulement de quoi garder une tache pour nous
									sendbuffer[0] = sendable;
									sendbuffer[1] = nb_bloc_precedent+current_task_blocs-sendable;
									MPI_Send(sendbuffer, 2, MPI_INT, i, TAG_YEP, MPI_COMM_WORLD);
									current_task_blocs -= sendable;
									nb_blocs_restants -= sendable;
									MPI_Irecv(&dummy, 1, MPI_INT, i, TAG_NEED_DATA, MPI_COMM_WORLD, &recv_requests[i]); //On relance un recv, au cas où ce processus nous renvoi une demande
								}else
								{
									//On n'a plus de tache à envoyer, on envoi NOPE
									sendbuffer[0] = 0;
									sendbuffer[1] = 0;
									MPI_Isend(sendbuffer, 2, MPI_INT, i, TAG_NOPE, MPI_COMM_WORLD, &send_requests[i]);
									nope_sent[i]=1;
								}
							}
						}
					}
				}
				if(nb_blocs_restants <= 1) //On n'a plus de tache à envoyer, on envoi NOPE à tout le monde (dont on n'a pas déjà envoyé un NOPE) en préventif
				{
					sendbuffer[0] = 0;
					sendbuffer[1] = 0;
					for(int i=0; i<nbProcess; i++)
					{
						if(!nope_sent[i])
						{
							MPI_Isend(&sendbuffer, 2, MPI_INT,i, TAG_NOPE, MPI_COMM_WORLD, &send_requests[i]);
							nope_sent[i]=1;
						}
					}
				}
			}
			// pour chaque pixel de son bloc de données
			for(int k=b*SIZE_BLOCK ; k<(b+1)*SIZE_BLOCK ; k++) {
				
				// calcul des indices globaux x et y
				//x = (k + (nb_bloc_precedent * SIZE_BLOCK)) / w;
				//y = (k + (nb_bloc_precedent * SIZE_BLOCK)) % w;
				x = (k + (current_task_start * SIZE_BLOCK)) / w;
				y = (k + (current_task_start * SIZE_BLOCK)) % w;

				//for(int l=0 ; l<rank ; l++) printf("\t\t");
				//printf("(%d,%d)\n",x,y);


			//Pour chaque ligne
		//	for (int i = 0; i < h_local; i++) {

				//unsigned short PRNG_state[3] = {0, 0, i*i*i};
				unsigned short PRNG_state[3] = {0, 0, x*x*x};

				//Pour chaque colonne
		//		for (unsigned short j = 0; j < w; j++) {

					/* calcule la luminance d'un pixel, avec sur-échantillonnage 2x2 */
					double pixel_radiance[3] = {0, 0, 0};
					//Pour chaque ligne de sous pixel
					for (int sub_i = 0; sub_i < 2; sub_i++) {
						//Pour chaque colonne de sous-pixel
						for (int sub_j = 0; sub_j < 2; sub_j++) {
							double subpixel_radiance[3] = {0, 0, 0};

							/* simulation de monte-carlo : on effectue plein de lancers de rayons et on moyenne */
							for (int s = 0; s < samples; s++) { 
								/* tire un rayon aléatoire dans une zone de la caméra qui correspond à peu près au pixel à calculer */
								double r1 = 2 * erand48(PRNG_state);
								double dx = (r1 < 1) ? sqrt(r1) - 1 : 1 - sqrt(2 - r1); 
								double r2 = 2 * erand48(PRNG_state);
								double dy = (r2 < 1) ? sqrt(r2) - 1 : 1 - sqrt(2 - r2);
								double ray_direction[3];
								copy(camera_direction, ray_direction);

								// anciennes lignes  
								// axpy(((sub_i + .5 + dy) / 2 + i) / h - .5, cy, ray_direction);
								// axpy(((sub_j + .5 + dx) / 2 + j) / w - .5, cx, ray_direction);

								axpy(((sub_i + .5 + dy) / 2 + x) / h - .5, cy, ray_direction);
								axpy(((sub_j + .5 + dx) / 2 + y) / w - .5, cx, ray_direction);
								normalize(ray_direction);

								double ray_origin[3];
								copy(camera_position, ray_origin);
								axpy(140, ray_direction, ray_origin);
								
								/* estime la lumiance qui arrive sur la caméra par ce rayon */
								double sample_radiance[3];
								radiance(ray_origin, ray_direction, 0, PRNG_state, sample_radiance);
								/* fait la moyenne sur tous les rayons */
								axpy(1. / samples, sample_radiance, subpixel_radiance);
							}
							clamp(subpixel_radiance);
							/* fait la moyenne sur les 4 sous-pixels */
							axpy(0.25, subpixel_radiance, pixel_radiance);
						}
					}

					// ligne originale 
					// copy(pixel_radiance, image + 3 * ((h - 1 - i) * w + j)); // <-- retournement vertical

					// On gère le retournement vertical au moment de la sauvegarde de l'image car notre division 
					// en blocs ne tombre pas forcement sur la fin d'une ligne
					if(processing_local_blocs)
					{
						copy(pixel_radiance, image + 3*k);
					}else
					{
						if(rank != 0)
						{
							copy(pixel_radiance, image_sup+current_sup_offset*3);
							++current_sup_offset;
						}else
						{
							copy(pixel_radiance, image+(current_task_start*SIZE_BLOCK+k)*3);
						}
					}
					
					
					
		//		} // for j
		//	} // for i

			} // for k
		}// for b

		if(processing_local_blocs)
		{
			processing_local_blocs=0;
			int sendbuffer[2] = {0, 0};
			for(int i=0; i<nbProcess; i++) //On a traité tous les blocs locaux, on envoi NOPE aux processus qui n'en ont pas reçu
			{
				if(!nope_sent[i])
				{
					MPI_Isend(&sendbuffer, 2, MPI_INT,i, TAG_NOPE, MPI_COMM_WORLD, &send_requests[i]);
					nope_sent[i]=1;
				}
			}
		}
	}
	

	/* Fin du chronomètre pour chaque process*/
	clock_end = wtime();

	/* Affichage du temps d'execution */
	double diff = (clock_end - clock_begin);
	double sec;
	int min;
	min = diff / 60;
	sec = diff - 60*min;
	printf("Runtime execution process %d: %d min %f seconds\n",rank,min,sec);

	/* ************************************************************************************** */
	// TEST : Cette partie devra etre changee 
	/* ************************************************************************************** */

	for(int i=0; i<nbProcess;i++) //On fait un wait pour s'assurer que toutes les requetes asynchrones sont finies, au cas où
	{
		if(i!= rank)
		{
			MPI_Wait(send_requests[i], &status);
			MPI_Wait(recv_requests[i], &status);
		}
	}

	//MICKAEL ! A TOI DE JOUER SUR CETTE PARTIE ! TU AS INTERET A CE QUE ÇA MARCHE BIEN POUR LA RECUPERATION !

	int tag_data = 15;
	

	if (rank == 0) {
		int offset = 3*nb_bloc_local*SIZE_BLOCK;
		for(int source = 1 ; source < nbProcess ; source++) {
			MPI_Recv(image + offset , 3*nb_bloc_locaux[source]*SIZE_BLOCK , MPI_DOUBLE , source , tag_data , MPI_COMM_WORLD,&status);
			offset +=  3*nb_bloc_locaux[source]*SIZE_BLOCK;
		}
	}
	else {
		MPI_Send(image , 3*nb_bloc_local*SIZE_BLOCK , MPI_DOUBLE , 0 , tag_data , MPI_COMM_WORLD);
	}


	/* ************************************************************************************** */

	/* Stocke l'image dans un fichier au format NetPbm 
	* !!!!! L'image doit etre retourné verticalement !!!!!
	*/
	if(rank == 0) 
	{
		struct passwd *pass; 
		char nom_sortie[100] = "";
		char nom_rep[30] = "";

		pass = getpwuid(getuid()); 
		sprintf(nom_rep, "/tmp/%s", pass->pw_name);
		mkdir(nom_rep, S_IRWXU);
		sprintf(nom_sortie, "%s/image.ppm", nom_rep);
		
		FILE *f = fopen(nom_sortie, "w");
		fprintf(f, "P3\n%d %d\n%d\n", w, h, 255); 
		for (int i = 0; i < w * h; i++) {
			// ligne originale 
			// fprintf(f,"%d %d %d ", toInt(image[3 * i]), toInt(image[3 * i + 1]), toInt(image[3 * i + 2]));
			
			int p = (h - (i/w) - 1) * w + (i%w);

			fprintf(f,"%d %d %d ", toInt(image[3 * p]), toInt(image[3 * p + 1]), toInt(image[3 * p + 2]));
		}
	  		 
		fclose(f); 
	}

	/* Libération des ressources */
	free(image);
	free(nb_bloc_locaux);
	free(other_process_ranks);
	free(other_process_hasWork);
	free(tasks);
	free(recv_requests);
	free(send_requests);
	free(nope_sent);
	if(rank != 0) {
		free(image_sup);
		free(pixel_sup);
	}

	if(rank==0)
	{
		/* Fin du chronomètre pour le processus 0 (Pour avoir le temps total)*/
		clock_end = wtime();

		/* Affichage du temps d'execution */
		double diff = (clock_end - clock_begin);
		double sec;
		int min;
		min = diff / 60;
		sec = diff - 60*min;
		printf("Temps total : %d min %f seconds\n",min,sec);
	}

	MPI_Finalize();
	return 0;
}
