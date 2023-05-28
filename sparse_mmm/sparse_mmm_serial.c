//  gcc -O0 -fopenmp sparse_mmm_loop.c -o sparse_mmm_loop

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define percentZero 98
#define elementRange 20
#define MATLENGTH 30000
#define INCREMENT 2000

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

        add_entry(result, i, j, val);
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

int main()
{

  printf("Matrix dimensions: %dX%d\n", MATLENGTH, MATLENGTH);

  long int alloc_size = MATLENGTH;

  double wakeup_var = wakeup_delay();

  detect_threads_setting();

  printf("Generating matrices...\n");

  struct sparse_matrix *matrix_a = generate_sparse_matrix(MATLENGTH, 55);

  struct sparse_matrix *matrix_b = generate_sparse_matrix(MATLENGTH, 32);

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
  }

  printf("Wakeup time: %f\n", wakeup_var);

  return 0;
}
