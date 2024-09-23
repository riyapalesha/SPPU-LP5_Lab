/*
Assignment 2
Write a program to implement Parallel Bubble Sort and Merge sort using OpenMP. Use existing algorithms and measure the performance of sequential and parallel algorithms.
*/

#include<iostream>
#include<vector>
#include<omp.h>
#include<chrono>
using namespace std;
using namespace std::chrono;

void print(vector<int>l)
{
    cout<<"Sorted List -> ";
    for(int x : l)
    {
        cout<<x<<" ";
    }
    cout<<endl;
}
void bubbleSort(vector<int>l)
{
    for(int i=0;i<l.size();i++)
    {
        for(int j=0;j<l.size()-i-1;j++)
        {
            if(l[j]>l[j+1])
            {
                swap(l[j],l[j+1]);
            }
        }
    }
    print(l);
}
void parallel_bubbleSort(vector<int>l)
{
    for(int i=0;i<l.size();i++)
    {
        if(i%2==0)
        {
            #pragma omp parallel for
            for(int j=0;j<l.size()-1;j+=2)
            {
                if(l[j]>l[j+1])
                {
                    swap(l[j],l[j+1]);
                }
            }
        }
        else
        {
            #pragma omp parallel for
            for(int j=1;j<l.size()-1;j+=2)
            {
                if(l[j]>l[j+1])
                {
                    swap(l[j],l[j+1]);
                }
            }
        }
    }
    print(l);
}
void merge(vector<int>&list, int l, int m, int r)
{
    vector<int>temp(list.size());
    int p=0;
    int i=l;
    int j=m+1;
    while(i<=m && j<=r)
    {
        if(list[i]<list[j])
        {
            temp[p]=list[i];
            i++;
            p++;
        }
        else
        {
            temp[p]=list[j];
            j++;
            p++;
        }
    }
    while(i<=m)
    {
        temp[p]=list[i];
        i++;
        p++;
    }
    while(j<=r)
    {
        temp[p]=list[j];
        j++;
        p++;
    }
    p=0; 
    for(int i=l;i<=r;i++)
    {
        list[i]=temp[p];
        p++;
    }
}
void mergeSort(vector<int>&list, int l, int r)
{
    if(l<r)
    {
        int m=(l+r)/2;
        mergeSort(list,l,m);
        mergeSort(list,m+1,r);
        merge(list,l,m,r);
    }
}
void parallel_mergeSort(vector<int>&list, int l, int r)
{
    if(l<r)
    {
        int m=(l+r)/2;
        #pragma omp parallel for
        {
            #pragma omp parallel for
            parallel_mergeSort(list,l,m);
            #pragma omp parallel for
            parallel_mergeSort(list,m+1,r);
        }
        merge(list,l,m,r);
    }
}
class List
{
    int len;
    vector<int>list;
    public:
    List(int l)
    {
        len=l;
        list.resize(l);
    }
    void create()
    {
        int x;
        for(int i=0;i<len;i++)
        {
            cout<<"Element "<<i+1<<" : ";
            cin>>x;
            list[i]=x;
        }
    }
    void show()
    {
        cout<<"List -> ";
        for(int x : list)
        {
            cout<<x<<" ";
        }
        cout<<endl;
    }
    void sort()
    {
        cout<<"----- Bubble Sort -----"<<endl;
        auto start=high_resolution_clock::now();
        bubbleSort(list);
        auto stop=high_resolution_clock::now();
        auto duration=duration_cast<microseconds>(stop-start);
        cout<<"Duration : "<<duration.count()<<endl;

        cout<<"----- Parallel Bubble Sort -----"<<endl;
        start=high_resolution_clock::now();
        parallel_bubbleSort(list);
        stop=high_resolution_clock::now();
        duration=duration_cast<microseconds>(stop-start);
        cout<<"Duration : "<<duration.count()<<endl;

        cout<<"----- Merge Sort -----"<<endl;
        start=high_resolution_clock::now();
        vector<int>tlist=list;
        mergeSort(tlist,0,len-1);
        print(tlist);
        stop=high_resolution_clock::now();
        duration=duration_cast<microseconds>(stop-start);
        cout<<"Duration : "<<duration.count()<<endl;

        cout<<"----- Parallel Merge Sort -----"<<endl;
        start=high_resolution_clock::now();
        tlist=list;
        parallel_mergeSort(tlist,0,len-1);
        print(tlist);
        stop=high_resolution_clock::now();
        duration=duration_cast<microseconds>(stop-start);
        cout<<"Duration : "<<duration.count()<<endl;
    }
};
int main()
{
    int len;
    cout<<"Enter length : ";
    cin>>len;
    List list(len);
    list.create();
    list.show();
    list.sort();
    return 0;
}