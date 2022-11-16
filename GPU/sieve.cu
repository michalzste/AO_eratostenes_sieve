#include "stdio.h"

/**
 * @brief Funkcja alokuje ciągły obszar pamięci na karcie graficznej o podanym rozmiarze
 * 
 * @param items liczba elementów tablicy bitów, którą chcemy zaalokować
 * @param bytesPerBitarray liczba bajtów na tablicę bitów
 * @param bitarrays  liczba tablic bitów, które chcemy zaalokować
 * 
 * @return unsigned char* 
 */
inline unsigned char *createBitearrays(size_t items, size_t *bytesPerBitarray, size_t bitarrays) {
  *bytesPerBitarray = items / 8;
  unsigned char *bitarraysMem;
  cudaMalloc(&bitarraysMem, *bytesPerBitarray * bitarrays);
  return bitarraysMem;
}

int main() {
  printf("%d", 0);
  return 0;
}