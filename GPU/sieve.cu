#include <stdio.h>
#include <stdint.h>

#define BYTE unsigned char


/**
 * @brief Funkcja alokuje ciągły obszar pamięci na karcie graficznej o podanym rozmiarze
 * optymalniej jest zaalokować jeden ciągły obszar niż tysiace mniejszych
 * 
 * @param elements liczba elementów tablicy bitów, którą chcemy zaalokować
 * @param bytesPerBitarray liczba bajtów na tablicę bitów
 * @param bitarrays  liczba tablic bitów, które chcemy zaalokować
 * 
 * @return BYTE* 
 */
inline BYTE *createBitearrays(size_t elements, size_t *bytesPerBitarray, size_t bitarrays) {

  // dodajemy potencjalny bajt, gdy liczba bitów nie jest podzielna przez 8
  *bytesPerBitarray = elements / 8 + ((elements % 8) > 1);  
  BYTE *bitarraysMem;
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
__global__ void sieve_chunk_gpu(BYTE *isPrimeArrays, size_t isPrimeBytes,
                                uint64_t defaultPrimeCount, uint64_t *primeCounts,
                                uint64_t *seedPrimes, uint64_t seedCount,
                                uint64_t chunkSize, uint64_t chunkCount, uint64_t chunkOffset) {

  // index konkretnego wywołania
  uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (index >= chunkCount) return;

  //index przesunięcia tablicy
  uint64_t offSetIndex = index + chunkOffset;

  //nowy wskaźnik na główną tablicę z liczbami pierwszymi
  BYTE *isPrime = isPrimeArrays + index * isPrimeBytes;

  //zakresy początku oraz końca przesunięcia tablicy
  uint64_t low = (offSetIndex + 1) * chunkSize;
  uint64_t high = low + chunkSize;

  //sprawdzamy czy nasze wartości graniczne przesunięcia tablicy nie są liczbami parzystymi w takim wypadku inkrementujemy je 
  //(jest to dodatkowa optymalizacja ze względu na to że poza 2 żadana liczba parzysta nie jest liczbą pierszą)
  if(low % 2 == 0)
    low++;

  if(high % 2 == 0)
    high--;

  //zerujemy komórki w tablicy którą używa dany wątek  
  for(size_t i = 0; i < isPrimeBytes; i++)
    isPrime[i] = 0;

  uint64_t primeCount = defaultPrimeCount;
  //zaczynamy od 1 ponieważ pierwszą liczbą z tablicy, która zawiera liczby pierwsze jest 2 wszystkie jej wielokrotności pomijamy w celu optymalizacji
  for(size_t i = 1; i < seedCount; i++) {
    //szukamy najmniejszej wielokrotności liczby, która jest dolnym zakresem przeszukiwania
    uint64_t lowMultiple = (uint64_t) (floor((double)low / (double) seedPrimes[i]) * (double) seedPrimes[i] );

    //sprawdzamy czy wyliczona najmniejsza wielkrotność nie jest mniejsza od liczby która stanowi dolną granicę przeszukiwania
    if(lowMultiple < low) {
      lowMultiple += seedPrimes[i];
    }

    uint64_t j = lowMultiple;


    while (j <= high) {
      uint64_t j_idx = (j - low) / 2;
            size_t byte = j_idx / 8;
            size_t offset_mask = 1 << (7 - (j_idx % 8));
            uint64_t is_set = (isPrime[byte] & offset_mask) == 0;
            primeCount -= is_set;
            BYTE potential_new_value = isPrime[byte] | offset_mask;
            isPrime[byte] = is_set * potential_new_value + (1 - is_set) * isPrime[byte];
            j += seedPrimes[i] * 2;
    }

  }

      primeCounts[index] = primeCount;
}

int main() {
  printf("%d\n", 0);
  return 0;
}