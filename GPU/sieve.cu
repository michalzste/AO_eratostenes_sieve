#include <chrono>
#include <stddef.h>
#include <stdio.h>
#include <iostream>
#include <stdint.h>
#include <sys/time.h>

#define THREADS 256
#define BLOCKS 16

inline unsigned char bitarray_get(unsigned char *bitarray, size_t index)
{
  size_t byte = index / 8;
  size_t offset_mask = 1 << (7 - (index % 8));
  return bitarray[byte] & offset_mask;
}

void startingSieve(uint64_t upperBound, uint64_t **primes, uint64_t *primeCount)
{
  unsigned char *is_prime = (unsigned char *)calloc(upperBound / 8 + ((upperBound % 8) > 1), 1);
  ;
  uint64_t total_primes = upperBound - 2;
  for (uint64_t p = 2; (p * p) < upperBound; p++)
  {
    if (bitarray_get(is_prime, p) == 0)
    {
      for (uint64_t i = p * p; i < upperBound; i += p)
      {
        if (bitarray_get(is_prime, i) == 0)
        {
          total_primes--;
          size_t byte = i / 8;
          size_t offset_mask = 1 << (7 - (i % 8));
          is_prime[byte] |= offset_mask;
        }
      }
    }
  }
  *primeCount = total_primes;
  *primes = (uint64_t *)malloc(sizeof(uint64_t) * total_primes);
  size_t i = 0;
  for (uint64_t p = 2; p < upperBound; p++)
  {
    if (bitarray_get(is_prime, p) == 0)
      (*primes)[i++] = p;
  }

  free(is_prime);
}

/**
 * @brief Funkcja alokuje ciągły obszar pamięci na karcie graficznej o podanym rozmiarze
 * optymalniej jest zaalokować jeden ciągły obszar niż tysiace mniejszych
 *
 * @param elements liczba elementów tablicy bitów, którą chcemy zaalokować
 * @param bytesPerBitarray liczba bajtów na tablicę bitów
 * @param bitarrays  liczba tablic bitów, które chcemy zaalokować
 *
 * @return unsigned char*
 */
inline unsigned char *createBitearrays(size_t elements, size_t *bytesPerBitarray, size_t bitarrays)
{

  // dodajemy potencjalny bajt, gdy liczba bitów nie jest podzielna przez 8
  *bytesPerBitarray = elements / 8 + ((elements % 8) > 1);
  unsigned char *bitarraysMem;
  cudaMalloc(&bitarraysMem, *bytesPerBitarray * bitarrays);
  return bitarraysMem;
}

/**
 * @brief Funkcja uruchamiana przez każdy wątek do szukania liczb pierwszych
 * __global__ oznacza że funkcja zostanie uruchomiona na GPU
 *
 * @param isPrimeArrays wskaźnik na tablicę bitów zaalokowaną funckją createBitearrays do której będą zapisywane informacje
 * o liczbach pierwszych przez wiele wątków
 * @param isPrimeBytes wielkość tablic bitowych używanych w poszczególnych wątkach (w bajtach)
 * @param defaultPrimeCount makxymalna ilość liczb pierwszych w wątku
 * (przykładowa tablica ma wielkość 10, parzyste liczby nie mogą być pierwsze więc defaultPrimeCount będzie 5)
 * @ param primeCounts liczba znalezionych liczb pierwszych
 * @param chunkCount liczba wątków/wywołań
 * @param chunkOffset przesunięcie tablicy isPrimeArrays, o nowy obszar do przejrzenia, dla nowego wątku
 * @param chunkSize ilość liczb, które są przeglądane przez dany wątek
 * @param seedCount ilość liczb pierwszych
 * @return __global__
 */
__global__ void sieveChunk(unsigned char *isPrimeArrays, size_t isPrimeBytes,
                           uint64_t defaultPrimeCount, uint64_t *primeCounts,
                           uint64_t *seedPrimes, uint64_t seedCount,
                           uint64_t chunkSize, uint64_t chunkCount, uint64_t chunkOffset)
{

  // index konkretnego wywołania
  uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;

  if (index >= chunkCount)
    return;

  // index przesunięcia tablicy
  uint64_t offSetIndex = index + chunkOffset;

  // nowy wskaźnik na główną tablicę z liczbami pierwszymi
  unsigned char *isPrime = isPrimeArrays + index * isPrimeBytes;

  // zakresy początku oraz końca przesunięcia tablicy
  uint64_t low = (offSetIndex + 1) * chunkSize;
  uint64_t high = low + chunkSize;

  // sprawdzamy czy nasze wartości graniczne przesunięcia tablicy nie są liczbami parzystymi w takim wypadku inkrementujemy je
  //(jest to dodatkowa optymalizacja ze względu na to że poza 2 żadana liczba parzysta nie jest liczbą pierszą)
  if (low % 2 == 0)
    low++;

  if (high % 2 == 0)
    high--;

  // zerujemy komórki w tablicy którą używa dany wątek
  for (size_t i = 0; i < isPrimeBytes; i++)
    isPrime[i] = 0;

  uint64_t primeCount = defaultPrimeCount;
  // zaczynamy od 1 ponieważ pierwszą liczbą z tablicy, która zawiera liczby pierwsze jest 2 wszystkie jej wielokrotności pomijamy w celu optymalizacji
  for (size_t i = 1; i < seedCount; i++)
  {
    // szukamy najmniejszej wielokrotności liczby, która jest dolnym zakresem przeszukiwania
    uint64_t lowMultiple = (uint64_t)(floor((double)low / (double)seedPrimes[i]) * (double)seedPrimes[i]);

    // sprawdzamy czy wyliczona najmniejsza wielkrotność nie jest mniejsza od liczby która stanowi dolną granicę przeszukiwania
    if (lowMultiple < low)
    {
      lowMultiple += seedPrimes[i];
    }

    uint64_t j = lowMultiple;

    while (j <= high)
    {
      uint64_t j_idx = (j - low) / 2;
      size_t byte = j_idx / 8;
      size_t offset_mask = 1 << (7 - (j_idx % 8));
      unsigned char potential_new_value = isPrime[byte] | offset_mask;
      if ((isPrime[byte] & offset_mask) == 0)
      {
        primeCount--;
        isPrime[byte] = potential_new_value;
      }
      j += seedPrimes[i] * 2;
    }
  }

  primeCounts[index] = primeCount;
}

/**
 * @brief Funkcja która zarządza chunkami
 *
 * @param chunkSize rozmiar chunka
 * @param chunkCount liczba chunków
 * @param chunkOffset offset chunków
 * @param chunkPrimeCount max ilość liczb pierwszych w chanku
 * @param seedPrimes potencjalne liczby pierwsze
 * @param seedPrimeCount  ilość potencjalnych liczb pierwszych
 * @param chunkPrimeCounts tu zapisywana jest informacja ile liczb pierwszych udało się znaleźć
 * @param processedChunks dodatkowa informacja jak wiele chunków zostało zakończonych
 */
void processChunks(uint64_t chunkSize, uint64_t chunkCount, uint64_t chunkOffset, uint64_t chunkPrimeCount, 
                  uint64_t *seedPrimes, uint64_t seedPrimeCount, uint64_t *chunkPrimeCounts)
{
  // liczba chunków: liczba wątków * liczba bloków na wątek
  uint64_t kernelChunkCount = THREADS * BLOCKS;

  // liczba wywołań jądra
  uint64_t invocations = 1;

  // korekcja liczby wywołań
  if (chunkCount < kernelChunkCount)
    kernelChunkCount = chunkCount;
  else
    invocations = chunkCount / kernelChunkCount + ((chunkCount % kernelChunkCount) > 0);

  uint64_t *seedPrimesLocal;
  size_t seedPrimesSize = sizeof(uint64_t) * seedPrimeCount;

  // alokacja pamięci
  cudaMalloc(&seedPrimesLocal, seedPrimesSize);

  // alokacja pamięci na wynikową tablicę
  size_t isPrimeBytes;
  unsigned char *isPrimeArrays = createBitearrays(chunkPrimeCount, &isPrimeBytes, kernelChunkCount);

  // alokacja pamięci na liczbę znalezionych liczb pierwszych
  uint64_t *primeCounts;
  cudaMalloc(&primeCounts, sizeof(uint64_t) * kernelChunkCount);

  // liczba zakończonych chunków
  uint64_t totalChunksProcessed = 0;

  for (uint64_t i = 0; i < invocations; i++)
  {
    // przesunięcie, które zwiększa się wraz z liczbą wątków
    uint64_t offset = i * kernelChunkCount;
    // zmienna przechoująca liczbę pozostałych wątków
    uint64_t remainingChunks = chunkCount - offset;

    // warunek sprawdzający czy liczba pozostałych wąktów nie większa od liczby maksymalnej wątków
    if (remainingChunks > kernelChunkCount)
    {
      remainingChunks = kernelChunkCount;
    }

    offset += chunkOffset;

    // Wywołanie nowego wątku
    sieveChunk<<<THREADS, BLOCKS>>>(
        isPrimeArrays, isPrimeBytes,
        chunkPrimeCount, primeCounts,
        seedPrimes, seedPrimeCount,
        chunkSize, remainingChunks, offset);

    // skopiowanie pamięci do cpu
    cudaMemcpy(chunkPrimeCounts + totalChunksProcessed, primeCounts, sizeof(uint64_t) * remainingChunks, cudaMemcpyDeviceToHost);
    totalChunksProcessed += remainingChunks;
  }

  // zwolnienie pamięci 
  cudaFree(isPrimeArrays);
  cudaFree(seedPrimes);
}

int main()
{
  uint64_t count = 100;
  std::cin >> count;

  auto start = std::chrono::high_resolution_clock::now();

  // górna granica zadana
  uint64_t upperBound = count;
  // zakres pierwotnych liczb pierwszych
  uint64_t chunkSize = (uint64_t)sqrt((double)upperBound);
  // dopełnienie do najbliższej gornej granicy mającej calkowity pierwiastek
  if (chunkSize * chunkSize < upperBound)
  {
    chunkSize++;
    upperBound = chunkSize * chunkSize;
  }
  // wielkosc tablicy do liczenia liczb pierwszych wymaganych dla gpu
  uint64_t chunkCount = chunkSize - 1;
  // wielkosc segmentu dla gpu
  uint64_t chunkPrimeCount = chunkSize / 2 + chunkSize % 2;
  // liczba liczb pierwszych w segmencie
  uint64_t *chunkPrimeCounts;
  // liczby pierwsze z do sqrt(gorna granica)
  uint64_t *seedPrimes;
  // liczba liczb pierwszych do sqrt(gorna granica)
  uint64_t seedPrimeCount;
  // sprawdzone segmenty
  uint64_t totalChunksChecked;
  // liczba znalezionych liczb pierwszych
  uint64_t foundPrimes;

  if (upperBound < 50013184)
  { // umowna granica oplacalnosci inicjalizacji gpu
    // obliczenie za pomocą CPU, inicjalizacja obliczen na GPU zbyt kosztowna dla tego zakresu
    startingSieve(upperBound, &seedPrimes, &seedPrimeCount);
    foundPrimes = seedPrimeCount;
    free(seedPrimes);
  }
  else
  {
    std::cout << "Processing " << chunkCount << " chunks. Up to " << upperBound << std::endl;
    // obliczenie za pomocą CPU liczb pierwszych potrzebnych dla obliczen GPU
    startingSieve(chunkSize, &seedPrimes, &seedPrimeCount);

    chunkPrimeCounts = (uint64_t *)calloc(sizeof(uint64_t), chunkCount);
    // obliczenia na GPU
    processChunks(chunkSize, chunkCount, 0, chunkPrimeCount, seedPrimes, seedPrimeCount, chunkPrimeCounts);

    totalChunksChecked = 0;
    foundPrimes = seedPrimeCount;
    // zliczanie liczb pierwszych z segmentów
    for (size_t j = 0; j < chunkCount; j++)
    {
      foundPrimes += chunkPrimeCounts[j];
      totalChunksChecked++;
    }
  }

  auto stop = std::chrono::high_resolution_clock::now();
  std::cout << "amount of primes: " << foundPrimes << " found in time: " << std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count() << " ms" << std::endl;
  return 0;
}
