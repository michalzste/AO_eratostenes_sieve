# include <cstdlib>
# include <iostream>
# include <iomanip>
# include <omp.h>
# include <chrono>

using namespace std;
using namespace std::chrono;

// process only odd numbers of a specified block
int eratosthenesOddSingleBlock(const int from, const int to)
{
  const int memorySize = (to - from + 1) / 2;
  // initialize
  char* isPrime = new char[memorySize];
  for (int i = 0; i < memorySize; i++)
    isPrime[i] = 1;
  for (int i = 3; i*i <= to; i+=2)
  {
    // skip numbers before current slice
    int minJ = ((from+i-1)/i)*i;
    if (minJ < i*i)
      minJ = i*i;
    // start value must be odd
    if ((minJ & 1) == 0)
      minJ += i;
    // find all odd non-primes
    for (int j = minJ; j <= to; j += 2*i)
    {
      int index = j - from;
      isPrime[index/2] = 0;
    }
  }
  // count primes in this block
  int found = 0;
  for (int i = 0; i < memorySize; i++)
    found += isPrime[i];
  // 2 is not odd => include on demand
  if (from <= 2)
    found++;
  delete[] isPrime;
  return found;
}

int eratosthenesBlockwise(int lastNumber, int sliceSize, bool useOpenMP)
{
  // enable/disable OpenMP
  omp_set_num_threads(useOpenMP ? omp_get_num_procs() : 1);
  int found = 0;
  // each slices covers ["from" ... "to"], incl. "from" and "to"
#pragma omp parallel for reduction(+:found)
  for (int from = 2; from <= lastNumber; from += sliceSize)
  {
    int to = from + sliceSize;
    if (to > lastNumber)
      to = lastNumber;
    found += eratosthenesOddSingleBlock(from, to);
  }
  return found;
}

int main(int argc, char const *argv[])
{
    unsigned n;
    cin >> n;

    auto start = high_resolution_clock::now();
    unsigned primes =  eratosthenesBlockwise(n, 128*1024, true);
    auto stop = high_resolution_clock::now();
    cout << "amount of primes: " << primes << " found in time: " <<  duration_cast<milliseconds>(stop - start).count() << "ms" << endl;
    return 0;
}
