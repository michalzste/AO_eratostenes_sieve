#include <iostream>
#include <vector>
#include <math.h>
#include <chrono>
using namespace std;
using namespace std::chrono;

void mark_multiples_of(unsigned n, std::vector<bool> &arr)
{
    for (int i = n * 2u; i < arr.size(); i += n)
    {
        arr[i] = false;
    }
}

void print_primes(vector<bool> &marked_primes)
{
    for (unsigned i = 3u; i < marked_primes.size(); i++)
    {
        if (marked_primes[i] == true)
        {
            cout << i << endl;
        }
    }
}

vector<bool> primeSieve(unsigned n)
{
    vector<bool> is_prime(n, true);

    mark_multiples_of(2u, is_prime);

    unsigned end = (unsigned)floor(sqrt(n));
    for (unsigned i = 3u; i <= end; i += 2u)
    {
        if (is_prime[i])
        {
            mark_multiples_of(i, is_prime);
        }
    }
    return is_prime;
}

int main(int argc, char const *argv[])
{
    unsigned n;
    cin >> n;

    auto start = high_resolution_clock::now();
    vector<bool> markedPrimes =  primeSieve(n);
    auto stop = high_resolution_clock::now();
    vector<bool> primes;
    for (int i = 2;i<markedPrimes.size(); i++)
    {
        if(markedPrimes[i]) primes.push_back(i);
    }
    
    cout << "amount of primes: " << primes.size() << " found in time: " <<  duration_cast<milliseconds>(stop - start).count() << "ms" << endl;

    return 0;
}
