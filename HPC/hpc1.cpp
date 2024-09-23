/*
 Assignment 1
 Design and implement Parallel Breadth First Search and Depth First Search based on existing algorithms using OpenMP. Use a Tree or an undirected graph for BFS and DFS .
*/

#include<iostream>
#include<vector>
#include<stack>
#include<queue>
#include<omp.h>
#include<chrono>
using namespace std;
using namespace std::chrono;

void dfs(vector<vector<int>>g, int start, int vertices)
{
    vector<int>visited(vertices, false);
    stack<int>s;
    visited[start]=true;
    s.push(start);
    while(!s.empty())
    {
        int t=s.top();
        s.pop();
        cout<<t<<" ";
        for(int x : g[t])
        {
            if(visited[x]==false)
            {
                visited[x]=true;
                s.push(x);
            }
        }
    }
}
void parallel_dfs(vector<vector<int>>g, int start, int vertices)
{
    vector<int>visited(vertices, false);
    stack<int>s;
    visited[start]=true;
    s.push(start);
    while(!s.empty())
    {
        int t=s.top();
        s.pop();
        cout<<t<<" ";
        #pragma omp parallel for
        for(int x : g[t])
        {
            if(visited[x]==false)
            {
                visited[x]=true;
                s.push(x);
            }
        }
    }
}
void bfs(vector<vector<int>>g, int start, int vertices)
{
    vector<int>visited(vertices, false);
    queue<int>q;
    visited[start]=true;
    q.push(start);
    while(!q.empty())
    {
        int t=q.front();
        q.pop();
        cout<<t<<" ";
        for(int x : g[t])
        {
            if(visited[x]==false)
            {
                visited[x]=true;
                q.push(x);
            }
        }
    }
}
void parallel_bfs(vector<vector<int>>g, int start, int vertices)
{
    vector<int>visited(vertices, false);
    queue<int>q;
    visited[start]=true;
    q.push(start);
    while(!q.empty())
    {
        int t=q.front();
        q.pop();
        cout<<t<<" ";
        #pragma omp parallel for
        for(int x : g[t])
        {
            if(visited[x]==false)
            {
                visited[x]=true;
                q.push(x);
            }
        }
    }
}
class Graph
{
    int vertices, edges;
    vector<vector<int>>g;
    public:
    Graph(int v, int e)
    {
        vertices=v;
        edges=e;
        g.resize(v);
    }
    void insert(int s, int d)
    {
        g[s].push_back(d);
        g[d].push_back(s);
    }
    void create()
    {
        int s,d;
        for(int i=0;i<edges;i++)
        {
            cout<<"Enter source : ";
            cin>>s;
            cout<<"Enter destination : ";
            cin>>d;
            insert(s,d);
        }
    }
    void show()
    {
        for(int i=0;i<vertices;i++)
        {
            cout<<i<<" -> ";
            for(int x : g[i])
            {
                cout<<x<<" ";
            }
            cout<<endl;
        }
    }
    void search()
    {
        cout<<"----- DFS -----"<<endl;
        auto start=high_resolution_clock::now();
        dfs(g,0,vertices);
        auto stop=high_resolution_clock::now();
        auto duration=duration_cast<milliseconds>(stop-start);
        cout<<"\nDuration : "<<duration.count()<<endl;

        cout<<"----- Parallel DFS -----"<<endl;
        start=high_resolution_clock::now();
        parallel_dfs(g,0,vertices);
        stop=high_resolution_clock::now();
        duration=duration_cast<milliseconds>(stop-start);
        cout<<"\nDuration : "<<duration.count()<<endl;

        cout<<"----- BFS -----"<<endl;
        start=high_resolution_clock::now();
        bfs(g,0,vertices);
        stop=high_resolution_clock::now();
        duration=duration_cast<milliseconds>(stop-start);
        cout<<"\nDuration : "<<duration.count()<<endl;

        cout<<"----- Parallel BFS -----"<<endl;
        start=high_resolution_clock::now();
        parallel_bfs(g,0,vertices);
        stop=high_resolution_clock::now();
        duration=duration_cast<milliseconds>(stop-start);
        cout<<"\nDuration : "<<duration.count()<<endl;

    }
};
int main()
{
    int v,e;
    cout<<"Enter vertices : ";
    cin>>v;
    cout<<"Enter edges : ";
    cin>>e;
    Graph g(v,e);
    g.create();
    g.show();
    g.search();
    return 0;
}

