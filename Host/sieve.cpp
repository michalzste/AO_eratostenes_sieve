#include <iostream>
#include <vector>
#include <math.h>
using namespace std;

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
        // cout << marked_primes[i];
        if (marked_primes[i] == true)
        {
            cout << i << endl;
        }
    }
}

vector<bool> primeSieve(unsigned n)
{
    // initialize array
    vector<bool> is_prime(n, true);

    // Strike out the multiples of 2 so that
    // the following loop can be faster
    mark_multiples_of(2u, is_prime);

    // Strike out the multiples of the prime
    // number between 3 and end
    unsigned end = (unsigned)floor(sqrt(n));
    for (unsigned i = 3u; i <= end; i += 2u)
    {
        if (is_prime[i])
        {
            mark_multiples_of(i, is_prime);
        }
    }
    print_primes(is_prime);
    return is_prime;
}

int main(int argc, char const *argv[])
{
    unsigned n;
    cin >> n;

    primeSieve(n);

    return 0;
}
