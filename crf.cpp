#include <iostream>
#include <vector>
#include <dai/alldai.h>  // Include main libDAI header file
#include <dai/bp.h>
#include<boost/python.hpp>

#define PY_ARRAY_UNIQUE_SYMBOL PyArrayDaiCRF
#include <numpy/arrayobject.h>

using namespace dai;
using namespace std;

typedef vector<int> edge;

PyObject * crf_map(PyArrayObject* unaries, PyArrayObject* edges, double edge_strength) {
    // validate input
    if (PyArray_NDIM(unaries) != 2)
        throw runtime_error("Unaries must be 2d array.");
    if (PyArray_NDIM(edges) != 2)
        throw runtime_error("Edges must be 2d array.");
    if (PyArray_TYPE(edges) != PyArray_INT64)
        throw runtime_error("Edges must be long integers.");
    if (PyArray_TYPE(unaries) != PyArray_FLOAT64)
        throw runtime_error("Unaries must be double.");
        
    npy_intp* unaries_dims = PyArray_DIMS(unaries);
    npy_intp* edges_dims = PyArray_DIMS(edges);
    if (edges_dims[1] != 2)
        throw runtime_error("Edges must be of size n_edges x 2.");
    int n_vertices = unaries_dims[0];
    int n_states = unaries_dims[1];
    int n_edges = edges_dims[0];

    cout << "n_vertices: " << n_vertices << " n_states: " << n_states << " n_edges: " << n_edges << " edge strength: " << edge_strength << endl;

    vector<Var> vars;
    vector<Factor> factors;
    vars.reserve(n_vertices);

    // add variables
    for(size_t i = 0; i < n_vertices; i++)
        vars.push_back(Var(i, n_states));

    factors.reserve(n_edges + n_vertices);
    // add unary factors
    for(size_t i = 0; i < n_vertices; i++){
        Factor unary_factor(vars[i]);
        for(size_t j = 0; j < n_states; j++)
            unary_factor.set(j, *((double*)PyArray_GETPTR2(unaries, i, j)));
        factors.push_back(unary_factor);
    }
    for(size_t e = 0; e < n_edges; e++){
        int e0 = *((long*)PyArray_GETPTR2(edges, e, 0));
        int e1 = *((long*)PyArray_GETPTR2(edges, e, 1));
        Factor pairwise_factor = createFactorPotts(vars[e0], vars[e1], edge_strength);
        //for (size_t i = 0; i < n_states; i++)
            //for(size_t j = 0; j < n_states; j++)
                //pairwise_factor.set(i + n_states * j, i==j? edge_strength : 0);
        factors.push_back(pairwise_factor);
    }
    
    FactorGraph fg(factors);
    size_t maxiter = 10000;
    Real   tol = 1e-9;
    size_t verb = 1;

    // Store the constants in a PropertySet object
    PropertySet opts;
    opts.set("maxiter",maxiter);  // Maximum number of iterations
    opts.set("tol",tol);          // Tolerance for convergence
    opts.set("verbose",verb);     // Verbosity (amount of output generated)

    BP mp(fg, opts("updates",string("SEQRND"))("logdomain",false)("inference",string("MAXPROD"))("damping",string("0.1")));
    mp.init();
    mp.run();
    vector<size_t> mpstate = mp.findMaximum();
    
    //PyArrayObject * map = (PyArrayObject*) PyArray_SimpleNew(1, (npy_intp*)&n_vertices, PyArray_INT);
    PyObject * map = PyArray_SimpleNew(1, (npy_intp*)&n_vertices, PyArray_INT);
    for(size_t i = 0; i < n_vertices; i++){
        ((int*)PyArray_GETPTR1(map, i))[0] = mpstate[i];
    }
    return map;
}

void* extract_pyarray(PyObject* x)
{
	return x;
}

BOOST_PYTHON_MODULE(daicrf){
	boost::python::converter::registry::insert(
	    &extract_pyarray, boost::python::type_id<PyArrayObject>());
    boost::python::def("crf_map", crf_map);
    import_array();
}
