#include <iostream>
#include <vector>
#include <math.h>
#include <thread>

using namespace std;

void mark_multiples_of(unsigned n, vector<bool> &arr)
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

vector<bool> sieve_eratosthenes_seq(unsigned n)
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
    cout<<"XD";
    return is_prime;
}

vector<int> sieve_eratosthenes_parrarel(int n)
{
    if (n < 2u)
    {
        return {};
    }

    // Only the prime numbers <= sqrt(n) are
    // needed to find the other ones
    unsigned end = (unsigned)floor(sqrt(n));
    // Find the primes numbers <= sqrt(n) thanks
    // to a sequential sieve of Eratosthenes
    vector<bool> primes = sieve_eratosthenes_seq(end);
    vector<bool> is_prime(n + 1u);
    for (unsigned i = 0u; i < n + 1u; ++i)
    {
        is_prime[i] = 1;
    }
    vector<thread> threads;

    // Computes the number of primes numbers that will
    // be handled by each thread. This number depends on
    // the maximum number of concurrent threads allowed
    // by the implementation and on the total number of
    // elements in primes
    size_t nb_primes_per_thread =
        (size_t)(ceil(
            (float)(primes.size()) /
            (float)(thread::hardware_concurrency())));

    for (size_t first = 0u; first < primes.size(); first += nb_primes_per_thread)
    {
        unsigned last = min(first + nb_primes_per_thread, primes.size());
        // Spawn a thread to strike out the multiples
        // of the prime numbers corresponding to the
        // elements of primes between first and last
        threads.emplace_back(
            [&primes, &is_prime](int begin, int end)
            {
                for (size_t i = begin; i < end; ++i)
                {
                    auto prime = primes[i];
                    for (int n = prime * 2u; n < is_prime.size(); n += prime)
                    {
                        is_prime[n] = 0;
                    }
                }
            },
            first, last);
    }

    for (thread &thr : threads)
    {
        thr.join();
    }

    vector<int> res = {2u};
    for (int i = 3u; i < is_prime.size(); i += 2u)
    {
        if (is_prime[i])
        {
            res.push_back(i);
        }
    }
    return res;
}

int main(int argc, char const *argv[])
{
    unsigned n;
    cin >> n;

    sieve_eratosthenes_parrarel(n);

    return 0;
}