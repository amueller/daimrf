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

void validate_unaries_edges(PyArrayObject* unaries, PyArrayObject* edges)
{
   if (PyArray_NDIM(unaries) != 2)
       throw runtime_error("Unaries must be 2d array.");
   if (PyArray_NDIM(edges) != 2)
       throw runtime_error("Edges must be 2d array.");
   if (PyArray_TYPE(edges) != PyArray_INT64)
       throw runtime_error("Edges must be long integers.");
   if (PyArray_TYPE(unaries) != PyArray_FLOAT64)
       throw runtime_error("Unaries must be double.");

   if (PyArray_DIMS(edges)[1] != 2)
       throw runtime_error("Edges must be of size n_edges x 2.");
}

PyObject * mrf(PyArrayObject* unaries, PyArrayObject* edges, PyArrayObject* edge_potentials) {
    // validate input
    validate_unaries_edges(unaries, edges);
    if (PyArray_NDIM(edge_potentials) != 2)
        throw runtime_error("Edge potentials must be n_classes x n_classes");
   if (PyArray_TYPE(edge_potentials) != PyArray_FLOAT64)
       throw runtime_error("Edge potentials must be double.");

    npy_intp* unaries_dims = PyArray_DIMS(unaries);
    npy_intp* edges_dims = PyArray_DIMS(edges);
    npy_intp* edge_potential_dims = PyArray_DIMS(edge_potentials);
    int n_vertices = unaries_dims[0];
    int n_states = unaries_dims[1];
    int n_edges = edges_dims[0];
    if ((edge_potential_dims[0] != n_states) or (edge_potential_dims[1] != n_states))
        throw runtime_error("Edge potentials must be n_classes x n_classes");

    cout << "n_vertices: " << n_vertices << " n_states: " << n_states << " n_edges: " << n_edges << endl;

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
        Factor pairwise_factor(VarSet(vars[e0], vars[e1]));
        for (size_t i = 0; i < n_states; i++)
            for(size_t j = 0; j < n_states; j++){
                cout << i << " " << j << " " << *((double*)PyArray_GETPTR2(edge_potentials, i, j)) << endl;
                pairwise_factor.set(i + n_states * j, *((double*)PyArray_GETPTR2(edge_potentials, i, j)));
            }
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
    
    PyObject * map = PyArray_SimpleNew(1, (npy_intp*)&n_vertices, PyArray_INT);
    if (map == NULL)
        throw runtime_error("Could not allocate output array.");
    for(size_t i = 0; i < n_vertices; i++){
        ((int*)PyArray_GETPTR1(map, i))[0] = mpstate[i];
    }
    return map;
}


PyObject * potts_crf(PyArrayObject* unaries, PyArrayObject* edges, double edge_strength) {
    // validate input
    validate_unaries_edges(unaries, edges);
    npy_intp* unaries_dims = PyArray_DIMS(unaries);
    npy_intp* edges_dims = PyArray_DIMS(edges);
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
    
    PyObject * map = PyArray_SimpleNew(1, (npy_intp*)&n_vertices, PyArray_INT);
    if (map == NULL)
        throw runtime_error("Could not allocate output array.");
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
    boost::python::def("potts_crf", potts_crf);
    boost::python::def("mrf", mrf);
    import_array();
}
