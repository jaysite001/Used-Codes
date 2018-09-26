#ifdef ENABLE_SSE
#include <xmmintrin.h>
#endif

static inline float euclidean_baseline_float(const int n, const float* x, const float* y){

	float result = 0.f;

	for(int i = 0; i < n; ++i){

		const float num = x[i] - y[i];

		result += num * num;

	}

	return result;

}



#ifdef ENABLE_SSE
static inline float euclidean_intrinsic_float(int n, const float* x, const float* y){

	float result=0;

	__m128 euclidean = _mm_setzero_ps();

	for (; n>3; n-=4) {

		const __m128 a = _mm_loadu_ps(x);

		const __m128 b = _mm_loadu_ps(y);

		const __m128 a_minus_b = _mm_sub_ps(a,b);

		const __m128 a_minus_b_sq = _mm_mul_ps(a_minus_b, a_minus_b);

		euclidean = _mm_add_ps(euclidean, a_minus_b_sq);

		x+=4;

		y+=4;

	}

	const __m128 shuffle1 = _mm_shuffle_ps(euclidean, euclidean, _MM_SHUFFLE(1,0,3,2));

	const __m128 sum1 = _mm_add_ps(euclidean, shuffle1);

	const __m128 shuffle2 = _mm_shuffle_ps(sum1, sum1, _MM_SHUFFLE(2,3,0,1));

	const __m128 sum2 = _mm_add_ps(sum1, shuffle2);

	// with SSE3, we could use hadd_ps, but the difference is negligible 




	_mm_store_ss(&result,sum2);

	//    _mm_empty();

	if (n)

		result += euclidean_baseline_float(n, x, y);	// remaining 1-3 entries

	return result;

}
#endif

float euclidean_float(const int dim, const float* const x, const float* const y){

	float (*euclidean_float)(const int, const float*, const float*) = euclidean_baseline_float;

#ifdef ENABLE_SSE
	euclidean_float = euclidean_intrinsic_float;

#endif

	return euclidean_float(dim, x, y);

}


/*
//OpenMP is not supported in mac. So disable the following unused cold.
 
void euclidean_sym_float(const int dim, const int nx, const float* const x, 

	float* const K){

		float (*euclidean_float)(const int, const float*, const float*) = euclidean_baseline_float;

#ifdef __SSE__

		euclidean_float = euclidean_intrinsic_float;

#endif

#pragma omp parallel shared(K, x)

		{

#pragma omp for schedule(static)

			for(int i = 0; i < nx; ++i){

				K[i * nx + i] = 0;

				for(int j = 0; j < i; ++j){

					const float euclidean = (*euclidean_float)(dim, &x[i * dim], &x[j * dim]);

					K[i * nx + j] = euclidean;

					K[j * nx + i] = euclidean;

				}

			}

		}

}







void euclidean_nonsym_float(const int dim, const int nx, const float* const x, 

	const int ny, const float* const y, float* const K){

		float (*euclidean_float)(const int, const float*, const float*) = euclidean_baseline_float;

#ifdef __SSE__

		euclidean_float = euclidean_intrinsic_float;

#endif

#pragma omp parallel shared(K, x, y)

		{

#pragma omp for  schedule(static)

			for(int i = 0; i < nx; ++i){

				for(int j = 0; j < ny; ++j){

					float euclidean = (*euclidean_float)(dim, &x[(long long)(i) * dim], &y[(long long)(j) * dim]);

					K[i * ny + j] = euclidean;

				}

			}

		}

}







void euclidean_many2one_float(const int dim, const int nx, const float* const x,

	const float* const y, float* const K){

		float (*euclidean_float)(const int, const float*, const float*) = euclidean_baseline_float;

#ifdef __SSE__

		euclidean_float = euclidean_intrinsic_float;

#endif

#pragma omp parallel shared(K, x, y)

		{

#pragma omp for  schedule(static)

			for(int i = 0; i < nx; ++i){

				float euclidean = (*euclidean_float)(dim, &x[i * dim], y);

				K[i] = euclidean;

			}

		}

}

*/
