/* basé sur on smallpt, a Path Tracer by Kevin Beason, 2008
 *  	http://www.kevinbeason.com/smallpt/ 
 *
 * Converti en C et modifié par Charles Bouillaguet, 2019
 *
 * Pour des détails sur le processus de rendu, lire :
 * 	https://docs.google.com/open?id=0B8g97JkuSSBwUENiWTJXeGtTOHFmSm51UC01YWtCZw
 */

#define _XOPEN_SOURCE 500
#include <omp.h>
#include <mpi.h>
#include <math.h>   
#include <stdlib.h> 
#include <stdio.h>
#include <stdbool.h>
#include <string.h>
#include <sys/time.h>
#include <sys/stat.h>  /* pour mkdir    */ 
#include <unistd.h>    /* pour getuid   */
#include <sys/types.h> /* pour getpwuid */
#include <pwd.h>       /* pour getpwuid */
#include <time.h>
#include <unistd.h> // pour gethostname
#include <immintrin.h>

#define SIZE_BLOCK 32	// nombre de pixels contenus dans un bloc
#define NBTHREAD 4
#define SIZE_PIXEL 3*sizeof(double)

//TAGS
#define TAG_NEED_DATA 10
#define TAG_YEP 11
#define TAG_NOPE 12
#define TAG_TASK_INFO 13
#define TAG_TASK_DATA 14
#define TAG_DATA 42

enum Refl_t {DIFF, SPEC, REFR};   /* types de matériaux (DIFFuse, SPECular, REFRactive) */

typedef __m256d avx;

/************ Definition des fonctions simd ********************************/
#define al __attribute__((aligned(32)))

#define avx_load(x) _mm256_load_pd(x)

// Extrait la i-eme valeur du registre
#define avx_extract(x, i) (((double*)&x)[i])

// Definit un registre simd contenant {d,d,d,d}
#define avx_scalar(d) _mm256_set1_pd(d)

//definit un registre simd contenant {d0,d1,d2,0}
#define avx_set(x,d0,d1,d2) (x = _mm256_setr_pd (d0, d1, d2, 0))

// Definit un registre simd contenant {0,0,0,0}
#define avx_zero(x) (x = _mm256_setzero_pd())

// Definit un registre simd etant la copie de x
#define avx_copy(x, y) (y = _mm256_set_pd(avx_extract(x,3), avx_extract(x,2), avx_extract(x,1),avx_extract(x,0)))

// y += a*x
#define avx_axpy(a, x, y) (y = _mm256_fmadd_pd(avx_scalar(a), x, y))

// x *= a
#define avx_scal(a, x) (x = _mm256_mul_pd(avx_scalar(a), x))

// z = x*y
#define avx_mul(x, y, z) (z = _mm256_mul_pd(x,y))

// x[i] = x[i] > 1 ? 1 : x[i] < 0 : 0 : x[i]
#define avx_clamp(x) (x = _mm256_min_pd(_mm256_max_pd(x, _mm256_setzero_pd()), avx_scalar(1.0)))

// dot(a, b)
static inline double avx_dot(const avx a, const avx b)
{
	
	//avx temp;
	//avx_mul(a,b,temp);
	//return avx_extract(temp, 0) + avx_extract(temp, 1) + avx_extract(temp, 2);

	//Version scalaire utilisée ici, parce qu'elle est bien plus rapide
	const double *da = ((double*)&a), *db = ((double*)&b);
	return da[0]*db[0] + da[1]*db[1] + da[2]*db[2];
}

// norm(x)
#define avx_norm(x) (sqrt(avx_dot(x,x)))

// normalize(x)
#define avx_normalize(x) (avx_scal(1.0/avx_norm(x), x))

// cross(a,b,c)
// [c2, c0, c1] = [a0, a1, a2]*[b1, b2, b0] - [b0, b1, b2]*[a1, a2, a0]
#define avx_cross(a, b, c) (c = _mm256_permute4x64_pd(_mm256_fmsub_pd(a,_mm256_permute4x64_pd(b,201),_mm256_mul_pd(b,_mm256_permute4x64_pd(a,201))),201)) 
//#define avx_cross(a,b,c) { double *da = ((double*)&a), *db = ((double*)&b); c=_mm256_setr_pd (da[1]*db[2]-db[1]*da[2], da[2]*db[0]-db[2]*da[0], da[0]*db[1]-db[0]*da[1], 0);}

//Copie les 3 plus hauts doubles du registre avx2 à l'emplacement donne
#define avx_copy3(x, mem) (_mm256_maskstore_pd(mem, _mm256_set_epi64x(0,(__int64_t)0xFFFFFFFFFFFFFFFF,(__int64_t)0xFFFFFFFFFFFFFFFF,(__int64_t)0xFFFFFFFFFFFFFFFF), x)) 

/*
struct Sphere { 
	double radius; 
	double position[3] __attribute__((aligned(32)));
	double emission[3] __attribute__((aligned(32)));     // couleur émise (=source de lumière)
	double color[3] __attribute__((aligned(32)));        // couleur de l'objet RGB (diffusion, refraction, ...) 
	enum Refl_t refl;       // type de reflection
	double max_reflexivity;
};
*/

typedef struct Sphere { 
	double radius; 
	avx position;
	avx emission;     /* couleur émise (=source de lumière) */
	avx color;        /* couleur de l'objet RGB (diffusion, refraction, ...) */
	enum Refl_t refl;       /* type de reflection */
	double max_reflexivity;
} Sphere;

static const int KILL_DEPTH = 7;
static const int SPLIT_DEPTH = 4;

static void initSphere(Sphere* s, double radius, avx position, avx emission, avx color, enum Refl_t refl, double max_reflexivity)
{
	s->radius = radius;
	s->position = position;
	s->emission = emission;
	s->color = color;
	s->refl = refl;
	s->max_reflexivity = max_reflexivity;
}

/* la scène est composée uniquement de spheres */
Sphere spheres[15]; 
static void initAllSpheres()
{
	// radius position,                         emission,     color,              material 
	initSphere(&(spheres[0]),1e5,  _mm256_setr_pd(1e5+1,  40.8,       81.6 , 0),   		_mm256_setr_pd(0,0,0,0),          _mm256_setr_pd(.75,  .25,  .25, 0),  DIFF, -1); // Left 
	initSphere(&(spheres[1]),1e5,  _mm256_setr_pd(-1e5+99, 40.8,       81.6 , 0),      _mm256_setr_pd(0,0,0,0),          _mm256_setr_pd(.25,  .25,  .75, 0),  DIFF, -1); // Right 
	initSphere(&(spheres[2]),1e5,  _mm256_setr_pd(50,      40.8,       1e5, 0),       _mm256_setr_pd(0,0,0,0),           _mm256_setr_pd(.75,  .75,  .75, 0),  DIFF, -1); // Back 
	initSphere(&(spheres[3]),1e5,  _mm256_setr_pd(50,      40.8,      -1e5 + 170, 0), _mm256_setr_pd(0,0,0,0),           _mm256_setr_pd(0,0,0,0),                 DIFF, -1); // Front 
	initSphere(&(spheres[4]),1e5,  _mm256_setr_pd(50,      1e5,        81.6, 0),      _mm256_setr_pd(0,0,0,0),           _mm256_setr_pd(0.75, .75,  .75, 0),  DIFF, -1); // Bottom 
	initSphere(&(spheres[5]),1e5,  _mm256_setr_pd(50,     -1e5 + 81.6, 81.6, 0),      _mm256_setr_pd(0,0,0,0),           _mm256_setr_pd(0.75, .75,  .75, 0),  DIFF, -1); // Top 
	initSphere(&(spheres[6]),16.5, _mm256_setr_pd(40,      16.5,       47, 0),        _mm256_setr_pd(0,0,0,0),           _mm256_setr_pd(.999, .999, .999, 0), SPEC, -1); // Mirror 
	initSphere(&(spheres[7]),16.5, _mm256_setr_pd(73,      46.5,       88, 0),        _mm256_setr_pd(0,0,0,0),           _mm256_setr_pd(.999, .999, .999, 0), REFR, -1); // Glass 
	initSphere(&(spheres[8]),10,   _mm256_setr_pd(15,      45,         112, 0),       _mm256_setr_pd(0,0,0,0),           _mm256_setr_pd(.999, .999, .999, 0), DIFF, -1); // white ball
	initSphere(&(spheres[9]),15,   _mm256_setr_pd(16,      16,         130, 0),       _mm256_setr_pd(0,0,0,0),           _mm256_setr_pd(.999, .999, 0, 0),    REFR, -1); // big yellow glass
	initSphere(&(spheres[10]),7.5,  _mm256_setr_pd(40,      8,          120, 0),       _mm256_setr_pd(0,0,0,0),           _mm256_setr_pd(.999, .999, 0   , 0), REFR, -1); // small yellow glass middle
	initSphere(&(spheres[11]),8.5,  _mm256_setr_pd(60,      9,          110, 0),       _mm256_setr_pd(0,0,0,0),           _mm256_setr_pd(.999, .999, 0   , 0), REFR, -1); // small yellow glass right
	initSphere(&(spheres[12]),10,   _mm256_setr_pd(80,      12,         92, 0),        _mm256_setr_pd(0,0,0,0),           _mm256_setr_pd(0, .999, 0, 0),       DIFF, -1); // green ball
	initSphere(&(spheres[13]),600,  _mm256_setr_pd(50,      681.33,     81.6, 0),      _mm256_setr_pd(12, 12, 12,0), 			_mm256_setr_pd(0,0,0,0),             DIFF, -1);  // Light 
	initSphere(&(spheres[14]),5,    _mm256_setr_pd(50,      75,         81.6, 0),      _mm256_setr_pd(0,0,0,0),           _mm256_setr_pd(0, .682, .999, 0), 	 DIFF, -1); // occlusion, mirror 

}

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

/* ************************************************************************************************************** */

static inline void axpy(double alpha, const double *x, double *y)
{
 	for (int i = 0; i < 3; i++)
		y[i] += alpha * x[i];
}

/* ************************************************************************************************************** */

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
double avx_sphere_intersect(const struct Sphere *s, avx avx_ray_origin, avx avx_ray_direction)
{ 

	avx avx_op;

	// Solve t^2*d.d + 2*t*(o-p).d + (o-p).(o-p)-R^2 = 0 
	//avx avx_pos;
	//avx_set(avx_pos,s->position[0],s->position[1],s->position[2]);

	avx_copy(s->position , avx_op);

	avx_axpy(-1 , avx_ray_origin , avx_op);

	double eps = 1e-4;
	
	double b = avx_dot(avx_op, avx_ray_direction);

	double discriminant = b * b - avx_dot(avx_op, avx_op) + s->radius * s->radius;

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

/* ************************************************************************************************************************************************ */

// détermine si le rayon intersecte l'une des spere; si oui renvoie true et fixe t, id
bool avx_intersect(avx avx_ray_origin, avx avx_ray_direction, double *t, int *id)
{ 
	int n = sizeof(spheres) / sizeof(struct Sphere);
	double inf = 1e20; 
	*t = inf;
	for (int i = 0; i < n; i++) {
		double d = avx_sphere_intersect(&spheres[i], avx_ray_origin, avx_ray_direction);
		if ((d > 0) && (d < *t)) {
			*t = d;
			*id = i;
		} 
	}
	return *t < inf;
} 



/* ************************************************************************************************************************************************ */
/* calcule (dans out) la luminance reçue par la camera sur le rayon donné */
void avx_radiance(avx avx_ray_origin, avx avx_ray_direction, int depth, unsigned short *PRNG_state, avx *avx_out)
{ 
	int id = 0;                             // id de la sphère intersectée par le rayon
	double t;                               // distance à l'intersection

	if (!avx_intersect(avx_ray_origin, avx_ray_direction, &t, &id)) {   		
		avx_zero(*avx_out); // if miss, return black
		return; 
	}
	
	const struct Sphere *obj = &spheres[id];
	
	/* point d'intersection du rayon et de la sphère */
	avx avx_x;

	avx_copy(avx_ray_origin , avx_x);
	
	avx_axpy(t , avx_ray_direction , avx_x);
	
	/* vecteur normal à la sphere, au point d'intersection */
	avx avx_n;

	avx_copy(avx_x , avx_n);

	avx_axpy(-1 , obj->position , avx_n);

	avx_normalize(avx_n);
	
	/* vecteur normal, orienté dans le sens opposé au rayon 
	   (vers l'extérieur si le rayon entre, vers l'intérieur s'il sort) */
	avx avx_nl;

	avx_copy(avx_n , avx_nl);

	if (avx_dot(avx_n, avx_ray_direction) > 0) avx_scal(-1, avx_nl);	
	
	/* couleur de la sphere */
	avx avx_f;
	avx_copy(obj->color , avx_f);

	double p = obj->max_reflexivity;

	/* processus aléatoire : au-delà d'une certaine profondeur,
	   décide aléatoirement d'arrêter la récusion. Plus l'objet est
	   clair, plus le processus a de chance de continuer. */
	depth++;
	if (depth > KILL_DEPTH) {
		if (erand48(PRNG_state) < p) {
			avx_scal(1/p , avx_f);
		} else {
			avx_copy(obj->emission , *avx_out);

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
		
		/* vecteur normal */
		avx avx_w;

		avx_copy(avx_nl , avx_w);
		
	  /* u est orthogonal à w */
		avx avx_u;

		avx avx_uw;
		avx_zero(avx_uw);

		if (fabs(avx_extract(avx_w , 0)) > .1) {
				avx_extract(avx_uw , 1) = 1;
		}
		
		else {
			avx_extract(avx_uw , 0) = 1;
		}

		avx_cross(avx_uw , avx_w , avx_u);

		avx_normalize(avx_u);
		
    /* v est orthogonal à u et w */
		avx avx_v;

		avx_cross(avx_w , avx_u , avx_v);
		
	  /* d est le vecteur incident aléatoire, selon la bonne distribution */
		avx avx_d;

		avx_zero(avx_d);

		avx_axpy(cos(r1) * r2s , avx_u , avx_d);

		avx_axpy(sin(r1) * r2s , avx_v , avx_d);
		
		avx_axpy(sqrt(1 - r2) , avx_w , avx_d);
		
		avx_normalize(avx_d);
	
		/* calcule récursivement la luminance du rayon incident */
		avx avx_rec;
		
		avx_radiance(avx_x, avx_d, depth, PRNG_state, &avx_rec);

		/* pondère par la couleur de la sphère, prend en compte l'emissivité */
		avx_mul(avx_f , avx_rec , *avx_out);

		avx_axpy(1 , obj->emission , *avx_out);

		return;
	}

	/* dans les deux autres cas (réflection parfaite / refraction), on considère le rayon
	   réfléchi par la spère */

	avx avx_reflected_dir;

	avx_copy(avx_ray_direction , avx_reflected_dir);

	avx_axpy(-2 * avx_dot(avx_n , avx_ray_direction) , avx_n , avx_reflected_dir);

	/* cas de la reflection SPEculaire parfaire (==mirroir) */
	if (obj->refl == SPEC) { 
		avx avx_rec;

		/* calcule récursivement la luminance du rayon réflechi */
		avx_radiance(avx_x, avx_reflected_dir, depth, PRNG_state, &avx_rec);

		/* pondère par la couleur de la sphère, prend en compte l'emissivité */
		avx_mul(avx_f , avx_rec , *avx_out);

		avx_axpy(1 , obj->emission , *avx_out);

		return;
	}

	/* cas des surfaces diélectriques (==verre). Combinaison de réflection et de réfraction. */
	//bool into = dot(n, nl) > 0;      /* vient-il de l'extérieur ? */
	bool into = avx_dot(avx_n , avx_nl) > 0;
	double ddn = avx_dot(avx_ray_direction , avx_nl);

	double nc = 1;                   /* indice de réfraction de l'air */
	double nt = 1.5;                 /* indice de réfraction du verre */
	double nnt = into ? (nc / nt) : (nt / nc);
	
	/* si le rayon essaye de sortir de l'objet en verre avec un angle incident trop faible,
	   il rebondit entièrement */
	double cos2t = 1 - nnt * nnt * (1 - ddn * ddn);
	if (cos2t < 0) {
		avx avx_rec;

		/* calcule seulement le rayon réfléchi */
		avx_radiance(avx_x, avx_reflected_dir, depth, PRNG_state, &avx_rec);

		avx_mul(avx_f , avx_rec , *avx_out);

		avx_axpy(1 , obj->emission , *avx_out);

		return;
	}
	
	/* calcule la direction du rayon réfracté */
	avx avx_tdir;

	avx_zero(avx_tdir);

	avx_axpy(nnt , avx_ray_direction , avx_tdir);

	avx_axpy(-(into ? 1 : -1) * (ddn * nnt + sqrt(cos2t)), avx_n, avx_tdir);

	/* calcul de la réflectance (==fraction de la lumière réfléchie) */
	double a = nt - nc;
	double b = nt + nc;
	double R0 = a * a / (b * b);

	//double c = 1 - (into ? -ddn : dot(tdir, n));
	double c = 1 - (into ? -ddn : avx_dot(avx_tdir, avx_n));

	double Re = R0 + (1 - R0) * c * c * c * c * c;   /* réflectance */
	double Tr = 1 - Re;                              /* transmittance */
	
	/* au-dela d'une certaine profondeur, on choisit aléatoirement si
	   on calcule le rayon réfléchi ou bien le rayon réfracté. En dessous du
	   seuil, on calcule les deux. */
	avx avx_rec;

	if (depth > SPLIT_DEPTH) {
		double P = .25 + .5 * Re;             /* probabilité de réflection */
		if (erand48(PRNG_state) < P) {
			avx_radiance(avx_x, avx_reflected_dir, depth, PRNG_state, &avx_rec);

			double RP = Re / P;
			
			avx_scal(RP , avx_rec);

		} else {
			avx_radiance(avx_x, avx_tdir, depth, PRNG_state, &avx_rec);

			double TP = Tr / (1 - P); 
			
			avx_scal(TP, avx_rec);
		}
	} else {
		avx avx_rec_re, avx_rec_tr;

		avx_radiance(avx_x, avx_reflected_dir, depth, PRNG_state, &avx_rec_re);

		avx_radiance(avx_x, avx_tdir, depth, PRNG_state, &avx_rec_tr);

		avx_zero(avx_rec);

		avx_axpy(Re , avx_rec_re , avx_rec);

		avx_axpy(Tr , avx_rec_tr , avx_rec);
	}
	/* pondère, prend en compte la luminance */
	avx_mul(avx_f , avx_rec , *avx_out);

	avx_axpy(1 , obj->emission , *avx_out);

	return;
}

/* ************************************************************************************************************************************************ */


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
	double clock_begin, clock_end, work_begin, work_end;
	initAllSpheres();
	clock_begin = wtime();

	// Petit cas test (small, quick and dirty):
	int w = 320;
	int h = 200;
	int samples = 50;


/*
	// Gros cas test (big, slow and pretty):
	int w = 3840;
	int h = 2160;
	int samples = 100;
*/

/*	
	int w = 1920;
	int h = 1080;
	int samples = 100;
*/


	if (argc >= 2) 
		samples = atoi(argv[1]);

	if(argc >= 4) {
		w = atoi(argv[2]);
		h = atoi(argv[3]);
	}

	static const double CST = 0.5135;  /* ceci défini l'angle de vue */

	avx avx_camera_position;
	avx_set(avx_camera_position , 50, 52, 295.6);

	avx avx_camera_direction;
	avx_set(avx_camera_direction, 0, -0.042612, -1);

	avx_normalize(avx_camera_direction);
	
	/* incréments pour passer d'un pixel à l'autre */
	
	avx avx_cx;
	avx_set(avx_cx , w * CST / h, 0, 0);   
	
	avx avx_cy;

	avx_cross(avx_cx , avx_camera_direction , avx_cy);

	avx_normalize(avx_cy);

	avx_scal(CST , avx_cy);

	
	/* précalcule la norme infinie des couleurs */
	int n = sizeof(spheres) / sizeof(struct Sphere);
	for (int i = 0; i < n; i++) {
		double f[3] al;
		avx_copy3(spheres[i].color, f);
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

	omp_set_num_threads(NBTHREAD);

	char hostname[50];
	gethostname(hostname , 50);

		/* Initialisation des variables de division des données*/
	
	// nombre total de blocs 
	int nb_bloc = ceil((double)(w*h) / SIZE_BLOCK);

	// nombre de blocs par processus
	int nb_bloc_local = ceil(nb_bloc / nbProcess);

	// on equilibre les blocs entre les processus
	if(rank < nb_bloc - (nb_bloc_local * nbProcess)) {
		nb_bloc_local++;
	}

	if(rank == 0) {
		printf("Calcul d'une image de taile %d*%d pixels\n",h,w);
		printf("Pour chaque pixel : %d échantillons\n",samples);
		printf("nombre total de pixels : %d\n",w*h);
		printf("Taille de blocs utilisée : %d pixels\n",SIZE_BLOCK);
		printf("Nombre total de blocs : %d\n",nb_bloc);
		printf("Nombre de processus : %d\n",nbProcess);
		printf("Nombre de threads crée par processus : %d\n",NBTHREAD);
	}

	// tampon mémoire pour l'image
	double *image , *image_sup;

	// tableau pour contenir le nombre de blocs initialement affectés a chaque processus
	int *nb_bloc_locaux;

	// Taille en blocs d'imagesup
	int imageSupSize;

	//Nombre de blocs utilises d'imagesup
	int used_imagesup_blocs = 0;


		/* Allocation mémoire */

	if(rank == 0) {
		// allocation de l'image totale
		if(posix_memalign((void**)&image , 32 , nb_bloc * SIZE_BLOCK * SIZE_PIXEL)) { perror("rank 0 : Impossible d'allouer l'image "); exit(1); }

		// pas de blocs supplementaires
		imageSupSize=0;
		image_sup = NULL;
	}
	else {

		// allocation d'un tampon principal
		if(posix_memalign((void**)&image , 32 , nb_bloc_local * SIZE_BLOCK * SIZE_PIXEL)) { perror("Impossible d'allouer l'image"); exit(1); }

		// allocation d'un tampon supplémentaire 
		imageSupSize = nb_bloc_local;
		if(posix_memalign((void**)&image_sup , 32 , imageSupSize * SIZE_BLOCK * SIZE_PIXEL)) { perror("Impossible d'allouer l'image"); exit(1); }
	}

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

	printf("Processus %d : calcul %d blocs sur la machine %s ...\n",rank,nb_bloc_locaux[rank],hostname);

	// definition des indices dans l'image globale
	int x,y;

	int nb_bloc_precedent = 0;
	for(int l=0 ; l<rank ; l++) nb_bloc_precedent += nb_bloc_locaux[l];

	//Gestion des taches courantes
	int current_task_start = nb_bloc_precedent;
	int current_task_blocs = nb_bloc_local;
	short processing_local_blocs = 1; // true
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

	int current_bloc_offset=0;
	//int current_sup_offset=0; //Offset pour écrire les pixels dans imagesup
	

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

	work_begin = wtime();
	while(process_with_work_counter > 0 || nbProcess==1)
	{
		//Boucle de demande de taches
		if(!processing_local_blocs)
		{
			while(nbProcess != 1)
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

		if((process_with_work_counter == 0 && nbProcess != 1) || (nbProcess==1 && !processing_local_blocs))
		{
			break;
		}

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
							MPI_Test(recv_requests + i, &finished, &status); //On vérifie si on a reçu une demande
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
									nb_bloc_local -= sendable;
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
			#pragma omp parallel for private(x,y) schedule(dynamic,1)
			for(int k=b*SIZE_BLOCK ; k<(b+1)*SIZE_BLOCK ; k++) {
				
				// calcul des indices globaux x et y
				x = (k + (current_task_start * SIZE_BLOCK)) / w;
				y = (k + (current_task_start * SIZE_BLOCK)) % w;

				//unsigned short PRNG_state[3] = {0, 0, i*i*i};
				unsigned short PRNG_state[3] = {0, 0, x*x*x};

				/* calcule la luminance d'un pixel, avec sur-échantillonnage 2x2 */				
				avx avx_pixel_radiance;
				avx_zero(avx_pixel_radiance);

				// Deux boucles imbriquées pour les sous pixels
				for (int sub_i = 0; sub_i < 2; sub_i++) {
						for (int sub_j = 0; sub_j < 2; sub_j++) {
							
							avx avx_subpixel_radiance;
							avx_zero(avx_subpixel_radiance);

							/* simulation de monte-carlo : on effectue plein de lancers de rayons et on moyenne */
							for (int s = 0; s < samples; s++) { 
								/* tire un rayon aléatoire dans une zone de la caméra qui correspond à peu près au pixel à calculer */
								double r1 = 2 * erand48(PRNG_state);
								double dx = (r1 < 1) ? sqrt(r1) - 1 : 1 - sqrt(2 - r1); 
								double r2 = 2 * erand48(PRNG_state);
								double dy = (r2 < 1) ? sqrt(r2) - 1 : 1 - sqrt(2 - r2);
								
								avx avx_ray_direction;

								avx_copy(avx_camera_direction , avx_ray_direction);

								avx_axpy(((sub_i + .5 + dy) / 2 + x) / h - .5, avx_cy, avx_ray_direction);
								avx_axpy(((sub_j + .5 + dx) / 2 + y) / w - .5, avx_cx, avx_ray_direction);

								avx_normalize(avx_ray_direction);
								
								avx avx_ray_origin;

								avx_copy(avx_camera_position , avx_ray_origin);

								avx_axpy(140, avx_ray_direction , avx_ray_origin);
								
								/* estime la lumiance qui arrive sur la caméra par ce rayon */
								avx avx_sample_radiance;
								
								avx_radiance(avx_ray_origin, avx_ray_direction, 0, PRNG_state, &avx_sample_radiance);
	
								/* fait la moyenne sur tous les rayons */
								avx_axpy(1. / samples , avx_sample_radiance , avx_subpixel_radiance);
							}
							
							avx_clamp(avx_subpixel_radiance);
							
							/* fait la moyenne sur les 4 sous-pixels */
							avx_axpy(0.25, avx_subpixel_radiance , avx_pixel_radiance);
							
						}
					}
			
					// On gère le retournement vertical au moment de la sauvegarde de l'image car notre division 
					// en blocs ne tombre pas forcement sur la fin d'une ligne
					if(processing_local_blocs)
					{
						avx_copy3(avx_pixel_radiance , image + 3*k);
					}else
					{
						if(rank != 0)
						{
							avx_copy3(avx_pixel_radiance , image_sup+(current_bloc_offset*SIZE_BLOCK + k)*3);
						}else
						{
							avx_copy3(avx_pixel_radiance, image+(current_task_start*SIZE_BLOCK+k)*3);
						}
					}
			} // for k
		}// for b
		current_bloc_offset = used_imagesup_blocs;
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

	// Affichage du temps de calcul pur pour chaque processus
	work_end = wtime();
	double diff_work = (work_end - work_begin);
	double sec_work;
	int min_work;
	min_work = diff_work / 60;
	sec_work = diff_work - 60*min_work;
	printf("Temps de calcul pur processus %d: %d min %f seconds\n",rank,min_work,sec_work);


	for(int i=0; i<nbProcess;i++) //On fait un wait pour s'assurer que toutes les requetes asynchrones sont finies, au cas où
	{
		if(i!= rank)
		{
			MPI_Wait(send_requests + i, &status);
			MPI_Wait(recv_requests + i, &status);
		}
	}

	/* ************************************************************************************** */
	// Regroupement des données
	/* ************************************************************************************** */


	int task_info[2] = {0,0};
	MPI_Status status_data_sup;
	
	if (rank == 0) {
		// reception des données locales initiales de chaque processus
		// certains blocks sont peut etre vide mais on bouche les trous plus tard
		// on utilise le TAG_DATA 
	
		int offset = 3*nb_bloc_locaux[0]*SIZE_BLOCK;
		for(int source = 1 ; source < nbProcess ; source++) {
			MPI_Recv(image + offset , 3*nb_bloc_locaux[source]*SIZE_BLOCK , MPI_DOUBLE , source , TAG_DATA , MPI_COMM_WORLD,&status);
			offset +=  3*nb_bloc_locaux[source]*SIZE_BLOCK;
		}

		/* Première boucle de reception terminée */

		// Pour chaque processus, on récupère les données supplementaires calculées 
		for(int source = 1 ; source < nbProcess ; source++) {
			
			do {
					MPI_Recv(task_info, 2, MPI_INT, source, TAG_TASK_INFO, MPI_COMM_WORLD, &status_data_sup);
				
					// si TASK_INFO != (0,0)
					if(task_info[0]) {
						MPI_Recv(image + 3*task_info[1]*SIZE_BLOCK , 3*task_info[0]*SIZE_BLOCK , MPI_DOUBLE , source , TAG_TASK_DATA , MPI_COMM_WORLD , &status_data_sup);
					}

			} while (task_info[0]);
		
		}

	}

	// Processus != 0
	else {

		int nb_bloc_sup = 0;

		// envoi des données locales initales
		// Certains blocks ont potentiellement été calculés par d'autres processus
		// on utilise le TAG_DATA 
		MPI_Send(image , 3*nb_bloc_local*SIZE_BLOCK , MPI_DOUBLE , 0 , TAG_DATA , MPI_COMM_WORLD);

		// tant qu'il reste des tasks dans le bloc de données supplémentaires 
		while(tasks_offset > 1) {
				
				task_info[1] = tasks[--tasks_offset];
				task_info[0] = tasks[--tasks_offset];

				// envoie du numéro du block de départ de la tache
				// envoie du nombre de blocks de la tâche
				MPI_Send(task_info, 2, MPI_INT, 0, TAG_TASK_INFO, MPI_COMM_WORLD);

				MPI_Send(image_sup + 3*(used_imagesup_blocs - task_info[0])*SIZE_BLOCK ,  3*task_info[0]*SIZE_BLOCK , MPI_DOUBLE , 0 , TAG_TASK_DATA , MPI_COMM_WORLD);

				used_imagesup_blocs -= task_info[0]	;

				nb_bloc_sup += task_info[0];
		}

		// il ne reste plus de tasks à envoyer, on envoie (0,0) pour indiquer au processus 0 que c'est ok 
		task_info[0] = 0;
		task_info[1] = 0;
		MPI_Send(task_info, 2, MPI_INT, 0, TAG_TASK_INFO, MPI_COMM_WORLD);
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

		printf("Image sauvegardé dans '%s'\n",nom_sortie);
		
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
		printf("Temps total d'execution : %d min %f seconds\n",min,sec);
	}

	MPI_Finalize();
	return 0;
}
