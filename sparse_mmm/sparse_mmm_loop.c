//  gcc -O0 -fopenmp sparse_mmm_loop.c -o sparse_mmm_loop

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

#define percentZero 98
#define elementRange 20
#define MATLENGTH 30000
#define INCREMENT 2000

#define BSIZE 128

/* This define is only used if you do not set the environment variable
   OMP_NUM_THREADS as instructed above, and if OpenMP also does not
   automatically detect the hardware capabilities.

   If you have a machine with lots of cores, you may wish to test with
   more threads, but make sure you also include results for THREADS=4
   in your report. */
#define THREADS 4

void detect_threads_setting()
{
  long int i, ognt;
  char *env_ONT;

  /* Find out how many threads OpenMP thinks it is wants to use */
#pragma omp parallel for
  for (i = 0; i < 1; i++)
  {
    ognt = omp_get_num_threads();
  }

  printf("omp's default number of threads is %d\n", ognt);

  /* If this is illegal (0 or less), default to the "#define THREADS"
     value that is defined above */
  if (ognt <= 0)
  {
    if (THREADS != ognt)
    {
      printf("Overriding with #define THREADS value %d\n", THREADS);
      ognt = THREADS;
    }
  }

  omp_set_num_threads(ognt);

  /* Once again ask OpenMP how many threads it is going to use */
#pragma omp parallel for
  for (i = 0; i < 1; i++)
  {
    ognt = omp_get_num_threads();
  }
  printf("Using %d threads for OpenMP\n", ognt);
}

typedef double data_t;

/* Create abstract data type for matrix */
typedef struct
{
  long int rowlen;
  data_t *data;
} matrix_rec, *matrix_ptr;

/* -=-=-=-=- Time measurement by clock_gettime() -=-=-=-=- */
/*
  As described in the clock_gettime manpage (type "man clock_gettime" at the
  shell prompt), a "timespec" is a structure that looks like this:

        struct timespec {
          time_t   tv_sec;   // seconds
          long     tv_nsec;  // and nanoseconds
        };
 */

double interval(struct timespec start, struct timespec end)
{
  struct timespec temp;
  temp.tv_sec = end.tv_sec - start.tv_sec;
  temp.tv_nsec = end.tv_nsec - start.tv_nsec;
  if (temp.tv_nsec < 0)
  {
    temp.tv_sec = temp.tv_sec - 1;
    temp.tv_nsec = temp.tv_nsec + 1000000000;
  }
  return (((double)temp.tv_sec) + ((double)temp.tv_nsec) * 1.0e-9);
}
/*
     This method does not require adjusting a #define constant

  How to use this method:

      struct timespec time_start, time_stop;
      clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_start);
      // DO SOMETHING THAT TAKES TIME
      clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_stop);
      measurement = interval(time_start, time_stop);

 */

/* -=-=-=-=- End of time measurement declarations =-=-=-=- */

/* This routine "wastes" a little time to make sure the machine gets
   out of power-saving mode (800 MHz) and switches to normal speed. */
double wakeup_delay()
{
  double meas = 0;
  int i, j;
  struct timespec time_start, time_stop;
  double quasi_random = 0;
  clock_gettime(CLOCK_REALTIME, &time_start);
  j = 100;
  while (meas < 1.0)
  {
    for (i = 1; i < j; i++)
    {
      /* This iterative calculation uses a chaotic map function, specifically
         the complex quadratic map (as in Julia and Mandelbrot sets), which is
         unpredictable enough to prevent compiler optimisation. */
      quasi_random = quasi_random * quasi_random - 1.923432;
    }
    clock_gettime(CLOCK_REALTIME, &time_stop);
    meas = interval(time_start, time_stop);
    j *= 2; /* Twice as much delay next time, until we've taken 1 second */
  }
  return quasi_random;
}

/* Create matrix of specified length */
matrix_ptr new_matrix(long int rowlen)
{
  long int i;

  /* Allocate and declare header structure */
  matrix_ptr result = (matrix_ptr)malloc(sizeof(matrix_rec));
  if (!result)
    return NULL; /* Couldn't allocate storage */
  result->rowlen = rowlen;

  /* Allocate and declare array */
  if (rowlen > 0)
  {
    data_t *data = (data_t *)calloc(rowlen * rowlen, sizeof(data_t));
    if (!data)
    {
      free((void *)result);
      printf("COULD NOT ALLOCATE %ld BYTES STORAGE \n",
             rowlen * rowlen * sizeof(data_t));
      exit(-1);
    }
    result->data = data;
  }
  else
    result->data = NULL;

  return result;
}

/* Set row length of matrix */
int set_matrix_rowlen(matrix_ptr m, long int rowlen)
{
  m->rowlen = rowlen;
  return 1;
}

/* Return row length of matrix */
long int get_matrix_rowlen(matrix_ptr m)
{
  return m->rowlen;
}

/* initialize matrix */
int init_matrix(matrix_ptr m, long int rowlen)
{
  long int i;

  if (rowlen > 0)
  {
    m->rowlen = rowlen;
    for (i = 0; i < rowlen * rowlen; i++)
      m->data[i] = (data_t)(i);
    return 1;
  }
  else
    return 0;
}

/* initialize matrix */
int zero_matrix(matrix_ptr m, long int rowlen)
{
  long int i, j;

  if (rowlen > 0)
  {
    m->rowlen = rowlen;
    for (i = 0; i < rowlen * rowlen; i++)
    {
      m->data[i] = 0;
    }
    return 1;
  }
  else
    return 0;
}

data_t *get_matrix_start(matrix_ptr m)
{
  return m->data;
}

void print2Dmatrix(matrix_ptr mat)
{
  int i, j;

  data_t *mat0 = get_matrix_start(mat);
  long int row_length = get_matrix_rowlen(mat);

  for (i = 0; i < row_length; i++)
  {
    for (j = 0; j < row_length; j++)
    {
      printf("%.1f, ", mat0[i * row_length + j]);
    }
    printf("\n");
  }
}

int countNumPos2D(matrix_ptr mat)
{
  long int row_length = get_matrix_rowlen(mat);

  data_t *mat0 = get_matrix_start(mat);
  int i, j, count;

  for (i = 0; i < row_length; i++)
  {
    for (j = 0; j < row_length; j++)
    {
      if (mat0[i * row_length + j] > 0)
      {
        count++;
      }
    }
  }

  return count;
}

void mmm_kij(matrix_ptr a, matrix_ptr b, matrix_ptr c)
{
  long int i, j, k;
  long int get_matrix_rowlen(matrix_ptr m);
  data_t *get_matrix_start(matrix_ptr m);
  long int row_length = get_matrix_rowlen(a);
  data_t *a0 = get_matrix_start(a);
  data_t *b0 = get_matrix_start(b);
  data_t *c0 = get_matrix_start(c);
  data_t r;

  int numIters = 0;

  for (k = 0; k < row_length; k++)
  {
    for (i = 0; i < row_length; i++)
    {
      r = a0[i * row_length + k];
      for (j = 0; j < row_length; j++)
        c0[i * row_length + j] += r * b0[k * row_length + j];
      numIters++;
      // printf("%ld: %f\n", row_length, c0[i*row_length+j]);
    }
  }
  printf("Num iters kij: %d\n", numIters);
}

// A sparse matrix is represented in three subarrays:
// - row_indices: The row indices of the non-zero entries.
// - column_indices: The column indices of the non-zero entries.
// - values: The values of the non-zero entries.
struct sparse_matrix
{
  int *row_indices;
  int *column_indices;
  double *values;
  int num_non_zeros;
};

// This function creates a new sparse matrix.
struct sparse_matrix *new_sparse_matrix()
{
  struct sparse_matrix *matrix = malloc(sizeof(struct sparse_matrix));
  matrix->row_indices = NULL;
  matrix->column_indices = NULL;
  matrix->values = NULL;
  matrix->num_non_zeros = 0;
  return matrix;
}

// This function adds a new entry to a sparse matrix.

void add_entry(struct sparse_matrix *matrix, int row, int column, double value)
{

  matrix->num_non_zeros++;
  matrix->row_indices = realloc(matrix->row_indices, sizeof(int) * matrix->num_non_zeros);
  matrix->column_indices = realloc(matrix->column_indices, sizeof(int) * matrix->num_non_zeros);
  matrix->values = realloc(matrix->values, sizeof(double) * matrix->num_non_zeros);
  matrix->row_indices[matrix->num_non_zeros - 1] = row;
  matrix->column_indices[matrix->num_non_zeros - 1] = column;
  matrix->values[matrix->num_non_zeros - 1] = value;

  // if (matrix->num_non_zeros % 1000 == 0)
  // {
  //   printf("Num entries generated: %d\n", matrix->num_non_zeros);
  // }
}
/*
struct sparse_matrix *new_sparse_matrix_malloc(int size)
{
  struct sparse_matrix *matrix = malloc(sizeof(struct sparse_matrix));
  matrix->row_indices = malloc(sizeof(int) * size / 2);
  matrix->column_indices = malloc(sizeof(int) * size / 2);
  matrix->values = malloc(sizeof(double) * size / 2);
  matrix->num_non_zeros = 0;
  return matrix;
}

// This function adds a new entry to a sparse matrix.

void add_entry_malloc(struct sparse_matrix *matrix, int row, int column, double value)
{

  matrix->num_non_zeros++;
  matrix->row_indices[matrix->num_non_zeros - 1] = row;
  matrix->column_indices[matrix->num_non_zeros - 1] = column;
  matrix->values[matrix->num_non_zeros - 1] = value;

  if (matrix->num_non_zeros % 1000 == 0)
  {
    printf("Num entries generated: %d\n", matrix->num_non_zeros);
  }
}
*/
/*
struct sparse_matrix *combine_same_elements(struct sparse_matrix *matrix_a)
{

  int maxRow = matrix_a->row_indices[matrix_a->num_non_zeros - 1];
  int maxCol = matrix_a->column_indices[matrix_a->num_non_zeros - 1];
  // printf("row: %d, col: %d\n", maxRow, maxCol);
  int numIters = 0;
  double accum;
  struct sparse_matrix *result = new_sparse_matrix();
  int i, j, k;
  for (i = 0; i <= maxRow; i++)
  {
    for (j = 0; j <= maxCol; j++)
    {
      accum = 0;
      for (k = 0; k < matrix_a->num_non_zeros; k++)
      {
        // printf("row_indices[k] : %d column_indices[k] == k : %d,\n i = %d j = %d\n",matrix_a -> row_indices[k], matrix_a -> column_indices[k], i, j);
        if (matrix_a->row_indices[k] == i && matrix_a->column_indices[k] == j)
        {
          accum += matrix_a->values[k];
        }
        numIters++;
      }
      if (accum > 0)
      {
        add_entry(result, i, j, accum);
        // printf("accum: %f\n", accum);
      }
    }
  }
  printf("Combine elements finishes \n");
  printf("NUM ITERS combine elements: %d\n, ", numIters);
  return result;
}

// This function multiplies two sparse matrices.
struct sparse_matrix *multiply_sparse_matrices(struct sparse_matrix *matrix_a, struct sparse_matrix *matrix_b)
{
  int numIters = 0;
  struct sparse_matrix *result = new_sparse_matrix();
  int i, j;

  for (i = 0; i < matrix_a->num_non_zeros; i++)
  {
    for (j = 0; j < matrix_b->num_non_zeros; j++)
    {
      if (matrix_a->column_indices[i] == matrix_b->row_indices[j])
      {
        add_entry(result, matrix_a->row_indices[i], matrix_b->column_indices[j], matrix_a->values[i] * matrix_b->values[j]);
      }
      numIters++;
    }
  }

  struct sparse_matrix *final_result = combine_same_elements(result);
  printf("NUM ITERS MMM: %d, \n", numIters);
  return final_result;
}
*/

struct sparse_matrix *multiply_sparse_matrices2(struct sparse_matrix *matrix_a, struct sparse_matrix *matrix_b, int size)
{
  struct sparse_matrix *result = new_sparse_matrix();
  int exists;
  int numIters = 0;
  int i, j, k;

  for (i = 0; i < size; i++)
  {
    for (j = 0; j < size; j++)
    {
      if (matrix_a->column_indices[i] == matrix_b->row_indices[j])
      { //  Element from A and B should be multiplied
        exists = 0;
        for (k = 0; k < result->num_non_zeros; k++)
        {
          if (result->row_indices[k] == matrix_a->row_indices[i] && result->column_indices[k] == matrix_b->column_indices[j])
          {
            result->values[k] += matrix_a->values[i] * matrix_b->values[j];
            exists = 1;
            break;
          }
        }
        if (!exists)
          add_entry(result, matrix_a->row_indices[i], matrix_b->column_indices[j], matrix_a->values[i] * matrix_b->values[j]);
      }
    }
  }
  return result;
}

struct sparse_matrix *multiply_sparse_matrices_blocking(struct sparse_matrix *matrix_a, struct sparse_matrix *matrix_b, int size)
{
  struct sparse_matrix *result = new_sparse_matrix();
  int exists = 0;
  int numIters = 0;
  int i, j, k, kk, jj, ii;

  int en = BSIZE * ((matrix_a->num_non_zeros) / BSIZE);

  for (jj = 0; jj < size; jj += BSIZE)
  {
    for (j = jj; j < jj + BSIZE && j < size; j++)
    {
      for (i = ii; i < ii + BSIZE && i < size; i++)
      {
        if (matrix_a->column_indices[i] == matrix_b->row_indices[j])
        {
          exists = 0;
          for (k = 0; k < result->num_non_zeros; k++)
          {
            if (result->row_indices[k] == matrix_a->row_indices[i] && result->column_indices[k] == matrix_b->column_indices[j])
            {
              result->values[k] += matrix_a->values[i] * matrix_b->values[j];
              exists = 1;
              break;
            }
          }
          if (!exists)
            add_entry(result, matrix_a->row_indices[i], matrix_b->column_indices[j], matrix_a->values[i] * matrix_b->values[j]);
        }
      }
    }
  }
  return result;
}

struct sparse_matrix *multiply_sparse_matrices_omp(struct sparse_matrix *matrix_a, struct sparse_matrix *matrix_b, int size)
{
  struct sparse_matrix *result = new_sparse_matrix();
  int exists = 0;
  int numIters = 0;
  int i, j, k;

  int *outputRow;
  int *outputCol;
  int *value;

  int totalValues = 0;

  // #pragma omp parallel shared(matrix_a -> column_indices, matrix_b->column_indices, result->column_indices, matrix_a->row_indices, matrix_b->row_indices, result->row_indices)
  {

    for (i = 0; i < size; i++)
    {
#pragma omp parallel for shared(result, matrix_a, matrix_b) private(j, k)
      for (j = 0; j < size; j++)
      {
        if (matrix_a->column_indices[i] == matrix_b->row_indices[j])
        {
          exists = 0;

          for (k = 0; k < result->num_non_zeros; k++)
          {
            if (result->row_indices[k] == matrix_a->row_indices[i] && result->column_indices[k] == matrix_b->column_indices[j])
            {
              result->values[k] += matrix_a->values[i] * matrix_b->values[j];
              exists = 1;
            }
            // numIters++;
          }

          if (!exists)
            add_entry(result, matrix_a->row_indices[i], matrix_b->column_indices[j], matrix_a->values[i] * matrix_b->values[j]);
        }
      }
    }
  }
  // printf("NUM ITERS MMM: %d, \n", numIters);
  return result;
}

void multiply_sparse_matrices_2d_output(struct sparse_matrix *matrix_a, struct sparse_matrix *matrix_b, matrix_ptr output, int length, int size)
{
  int exists;
  int numIters = 0;
  int i, j, k;

  for (i = 0; i < size; i++)
  {
    for (j = 0; j < size; j++)
    {
      if (matrix_a->column_indices[i] == matrix_b->row_indices[j])
      {
        output->data[(matrix_a->row_indices[i]) * length + (matrix_b->column_indices[j])] += matrix_a->values[i] * matrix_b->values[j];
      }
    }
  }
}

int multiply_sparse_matrices_2d_output_omp(struct sparse_matrix *matrix_a, struct sparse_matrix *matrix_b, matrix_ptr output, int length, int size)
{
  int exists;
  int numIters = 0;
  int i, j;

#pragma omp parallel for shared(output, matrix_a, matrix_b) private(i, j)
  for (i = 0; i < size; i++)
  {
    for (j = 0; j < size; j++)
    {
      if (matrix_a->column_indices[i] == matrix_b->row_indices[j])
      {
        output->data[(matrix_a->row_indices[i]) * length + (matrix_b->column_indices[j])] += (matrix_a->values[i]) * (matrix_b->values[j]);
      }
    }
  }
  return numIters;
}

struct sparse_matrix *generate_sparse_matrix(long int size, int seed)
{
  struct sparse_matrix *result = new_sparse_matrix();

  srand(seed);
  double val;
  int i, j;

  for (i = 0; i < size; i++)
  {
    for (j = 0; j < size; j++)
    {
      if (rand() % size == 1)
      {

        val = (rand() % elementRange) + 1;

        // val = 2;

        add_entry(result, i, j, val);
        // m->data[i * size + j] = val;
        //  printf("M data: %f\n",m -> data[i*size + j]);
      }
    }
  }

  return result;
}

struct sparse_matrix *generate_both_matrices(long int size, int seed, matrix_ptr m)
{
  struct sparse_matrix *result = new_sparse_matrix();
  double val;

  srand(seed);
  int i, j;

  for (i = 0; i < size; i++)
  {
    for (j = 0; j < size; j++)
    {
      if (rand() % size == 1)
      {

        val = rand() % elementRange;
        add_entry(result, i, j, val);
        m->data[i * size + j] = val;
        // printf("M data: %f\n",m -> data[i*size + j]);
      }
      else
      {
        m->data[i * size + j] = 0;
      }
    }
  }

  return result;
}

void printMatrix(struct sparse_matrix *result)
{
  int i;
  for (i = 0; i < result->num_non_zeros; i++)
  {
    printf("(%d, %d, %.1lf)  \n", result->row_indices[i], result->column_indices[i], result->values[i]);
  }
}

void checkResults(struct sparse_matrix *result, matrix_ptr m, long int size)
{
  long int numDiff = 0;
  int i, j, k;

  for (i = 0; i < m->rowlen; i++)
  {

    for (j = 0; j < m->rowlen; j++)
    {

      if (m->data[i * m->rowlen + j] > 0)
      {
        for (k = 0; k < result->num_non_zeros; k++)
        {
          if (m->data[i * m->rowlen + j] == result->values[k] && result->row_indices[k] == i && result->column_indices[k] == j)
          {
            continue;
          }
          else if (k + 1 == result->num_non_zeros)
          {
            numDiff++;
          }
        }
      }
    }

    printf("Number different: %ld: \n", numDiff);

    // if (m->data[result->row_indices[i] * size + result->column_indices[i]] != result->values[i])
    // {
    //   printf("Regular matrix val: %f\t Sparse Matrix Val: %f\n", m->data[result->row_indices[i] * size + result->column_indices[i]], result->values[i]);
    //   numDiff++;
    // }
  }

  if (numDiff > 0)
  {
    printf("Results do not match, number of different elements: %ld\n", numDiff);
  }
  else
  {
    printf("Results match for both matrices.\n");
  }
}

void checkResultsNew(struct sparse_matrix *result, matrix_ptr m, long int size)
{
  int numDiff = 0;
  int i;
  for (i = 0; i < result->num_non_zeros; i++)
  {
    if (m->data[(result->row_indices[i]) * size + result->column_indices[i]] != result->values[i])
    {
      numDiff++;
    }
  }

  printf("Num diff: %d\n", numDiff);
}

int main()
{

  printf("Matrix dimensions: %dX%d\n", MATLENGTH, MATLENGTH);

  long int alloc_size = MATLENGTH;

  // matrix_ptr a0 = new_matrix(alloc_size);

  // matrix_ptr b0 = new_matrix(alloc_size);

  // matrix_ptr c0 = new_matrix(alloc_size);

  matrix_ptr result_mat = new_matrix(alloc_size);

  double wakeup_var = wakeup_delay();

  detect_threads_setting();

  printf("Generating matrices...\n");

  struct sparse_matrix *matrix_a = generate_sparse_matrix(MATLENGTH, 55);

  struct sparse_matrix *matrix_b = generate_sparse_matrix(MATLENGTH, 32);

  // struct sparse_matrix *matrix_a = generate_both_matrices(MATLENGTH, 55, a0);

  // struct sparse_matrix *matrix_b = generate_both_matrices(MATLENGTH, 32, b0);

  printf("MATRIX A with number of elements %d\n", matrix_a->num_non_zeros);
  // printMatrix(matrix_a);

  printf("MATRIX B with number of elements %d \n", matrix_b->num_non_zeros);

  // printMatrix(matrix_b);

  double measurement;

  struct timespec time_start, time_stop;
  /*
    clock_gettime(CLOCK_REALTIME, &time_start);

    // DO SOMETHING THAT TAKES TIME
    struct sparse_matrix *result = multiply_sparse_matrices(matrix_a, matrix_b);

    clock_gettime(CLOCK_REALTIME, &time_stop);
    measurement = interval(time_start, time_stop);

    printMatrix(result);

    printf("Time of sparse MMM: %f\n\n", measurement);


  */
  int i;

  for (i = INCREMENT; i < MATLENGTH; i += INCREMENT)
  {

    clock_gettime(CLOCK_REALTIME, &time_start);

    // DO SOMETHING THAT TAKES TIME
    struct sparse_matrix *result2 = multiply_sparse_matrices2(matrix_a, matrix_b, i);

    clock_gettime(CLOCK_REALTIME, &time_stop);
    measurement = interval(time_start, time_stop);

    // printMatrix(result2);

    printf("Time of sparse MMM: %f\n\n", measurement);

    clock_gettime(CLOCK_REALTIME, &time_start);

    // DO SOMETHING THAT TAKES TIME
    struct sparse_matrix *result_blocking = multiply_sparse_matrices_blocking(matrix_a, matrix_b, i);

    clock_gettime(CLOCK_REALTIME, &time_stop);
    measurement = interval(time_start, time_stop);

    // printMatrix(result_blocking);

    printf("Time of sparse MMM blocking: %f\n\n", measurement);

    // clock_gettime(CLOCK_REALTIME, &time_start);

    // // DO SOMETHING THAT TAKES TIME
    // struct sparse_matrix *result_omp = multiply_sparse_matrices_omp(matrix_a, matrix_b, i);

    // clock_gettime(CLOCK_REALTIME, &time_stop);
    // measurement = interval(time_start, time_stop);

    // printMatrix(result_omp);

    // printf("Time of sparse MMM openMP: %f\n\n", measurement);

    clock_gettime(CLOCK_REALTIME, &time_start);

    // DO SOMETHING THAT TAKES TIME
    multiply_sparse_matrices_2d_output(matrix_a, matrix_b, result_mat, MATLENGTH, i);

    clock_gettime(CLOCK_REALTIME, &time_stop);
    measurement = interval(time_start, time_stop);

    // print2Dmatrix(result_mat);

    printf("Time of sparse MMM with 2D output: %f\n\n", measurement);

    zero_matrix(result_mat, get_matrix_rowlen(result_mat));

    clock_gettime(CLOCK_REALTIME, &time_start);

    // DO SOMETHING THAT TAKES TIME
    int num_2D_omp;
    num_2D_omp = multiply_sparse_matrices_2d_output_omp(matrix_a, matrix_b, result_mat, MATLENGTH, i);

    // print2Dmatrix(result_mat);
    // printf("Num iters 2D OMP: %d", num_2D_omp);

    clock_gettime(CLOCK_REALTIME, &time_stop);
    measurement = interval(time_start, time_stop);

    printf("Time of sparse MMM with 2D output OMP: %f\n", measurement);

    int numPos2Domp = countNumPos2D(result_mat);

    zero_matrix(result_mat, get_matrix_rowlen(result_mat));

    // printf("\n\nNum values in result: %d\n", result->num_non_zeros);
    printf("\nNum values in result2: %d\n", result2->num_non_zeros);
    printf("\nNum values in result blocking: %d\n", result_blocking->num_non_zeros);

    printf("\nNum values in result 2D OMP: %d\n\n\n\n", numPos2Domp);
    // printf("\nNum values in result openMP: %d\n", result_omp->num_non_zeros);

    // clock_gettime(CLOCK_REALTIME, &time_start);

    // // DO SOMETHING THAT TAKES TIME
    // int numIters2Domp = multiply_sparse_matrices_2d_output_omp(matrix_a, matrix_b, result_mat, MATLENGTH, i);

    // clock_gettime(CLOCK_REALTIME, &time_stop);
    // measurement = interval(time_start, time_stop);

    // printf("Num iters 2D onenMP: %d\n", numIters2Domp);

    // printf("Time of sparse MMM with 2D output openMP: %f\n\n", measurement);

    // clock_gettime(CLOCK_REALTIME, &time_start);

    // // DO SOMETHING THAT TAKES TIME
    // mmm_kij(a0, b0, c0);

    // clock_gettime(CLOCK_REALTIME, &time_stop);
    // measurement = interval(time_start, time_stop);

    // printf("Time of mmm_kij: %f\n", measurement);
    /*
      int i, j;
      for (i = 0; i < MATLENGTH; i++)
      {
        for (j = 0; j < MATLENGTH; j++)
        {

          // printf("a0: %f,  b0: %f,  c0: %f \n", a0->data[i * MATLENGTH + j], b0->data[i * MATLENGTH + j], c0->data[i * MATLENGTH + j]);
          // if ((c0->data[i * MATLENGTH + j]) != (result_mat->data[i * MATLENGTH + j]))
          if ((c0->data[i * MATLENGTH + j]) > 0)
            printf("Not matching: %d, %f, %f ", (i * MATLENGTH + j), (c0->data[i * MATLENGTH + j]), (result_mat->data[i * MATLENGTH + j]));
        }
        printf("");
      }

      */

    // checkResultsNew(result2, c0, MATLENGTH);
  }

  printf("Wakeup time: %f\n", wakeup_var);

  return 0;
}
