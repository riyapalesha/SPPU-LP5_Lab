// Assignment 3
// Implement Min, Max, Sum and Average operations using Parallel Reduction.

#include<iostream>
#include<vector>
#include<climits>
#include<omp.h>
#include<chrono>
using namespace std;
using namespace std::chrono;

void min(vector<int>list)
{
    int ans=INT_MAX;
    auto start=high_resolution_clock::now();
    for(int x : list)
    {
        if(x<ans)
        {
            ans=x;
        }
    }
    auto stop=high_resolution_clock::now();
    auto duration=duration_cast<microseconds>(stop-start);
    cout<<"Ans : "<<ans<<" Duration : "<<duration.count()<<endl;

    ans=INT_MAX;
    start=high_resolution_clock::now();
    #pragma omp parallel for reduction(min : ans)
    for(int x : list)
    {
        if(x<ans)
        {
            ans=x;
        }
    }
    stop=high_resolution_clock::now();
    duration=duration_cast<microseconds>(stop-start);
    cout<<"Ans : "<<ans<<" Duration : "<<duration.count()<<endl;
}
void max(vector<int>list)
{
    int ans=INT_MIN;
    auto start=high_resolution_clock::now();
    for(int x : list)
    {
        if(x>ans)
        {
            ans=x;
        }
    }
    auto stop=high_resolution_clock::now();
    auto duration=duration_cast<microseconds>(stop-start);
    cout<<"Ans : "<<ans<<" Duration : "<<duration.count()<<endl;

    ans=INT_MIN;
    start=high_resolution_clock::now();
    #pragma omp parallel for reduction(max : ans)
    for(int x : list)
    {
        if(x>ans)
        {
            ans=x;
        }
    }
    stop=high_resolution_clock::now();
    duration=duration_cast<microseconds>(stop-start);
    cout<<"Ans : "<<ans<<" Duration : "<<duration.count()<<endl;
}
void sum(vector<int>list)
{
    int ans=0;
    auto start=high_resolution_clock::now();
    for(int x : list)
    {
        ans+=x;
    }
    auto stop=high_resolution_clock::now();
    auto duration=duration_cast<microseconds>(stop-start);
    cout<<"Ans : "<<ans<<" Duration : "<<duration.count()<<endl;

    ans=0;
    start=high_resolution_clock::now();
    #pragma omp parallel for reduction(+ : ans)
    for(int x : list)
    {
        ans+=x;
    }
    stop=high_resolution_clock::now();
    duration=duration_cast<microseconds>(stop-start);
    cout<<"Ans : "<<ans<<" Duration : "<<duration.count()<<endl;
}
void avg(vector<int>list)
{
    int ans=0;
    auto start=high_resolution_clock::now();
    for(int x : list)
    {
        ans+=x;
    }
    ans=ans/list.size();
    auto stop=high_resolution_clock::now();
    auto duration=duration_cast<microseconds>(stop-start);
    cout<<"Ans : "<<ans<<" Duration : "<<duration.count()<<endl;

    ans=0;
    start=high_resolution_clock::now();
    #pragma omp parallel for reduction(+ : ans)
    for(int x : list)
    {
        ans+=x;
    }
    ans=ans/list.size();
    stop=high_resolution_clock::now();
    duration=duration_cast<microseconds>(stop-start);
    cout<<"Ans : "<<ans<<" Duration : "<<duration.count()<<endl;
}

int main()
{
    int len;
    cout<<"Enter length : ";
    cin>>len;
    vector<int>list(len);
    int x;
    for(int i=0;i<len;i++)
    {
        cout<<"Element "<<i<<" : ";
        cin>>x;
        list[i]=x;
    }
    min(list);
    max(list);
    sum(list);
    avg(list);
}