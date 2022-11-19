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
 * @param primeCounts liczba znalezionych liczb pierwszych
 * @param chunkCount liczba wątków/wywołań
 * @return __global__ 
 */
__global__ void sieveChunk(BYTE *isPrimeArrays, size_t isPrimeBytes, uint64_t defaultPrimeCount, 
                           uint64_t *primeCounts, uint64_t chunkCount) {

  // index konkretnego wywołania
  uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (index >= chunkCount) return;
}

int main() {
  printf("%d\n", 0);
  return 0;
}