#include <iostream>
#include <vector>
#include <dai/alldai.h>  // Include main libDAI header file
#include <dai/bp.h>     // believe propagation
#include <dai/jtree.h>  // junction tree
#include <dai/gibbs.h>
#include <dai/treeep.h>
#include <dai/trwbp.h>
#include<boost/python.hpp>

#define PY_ARRAY_UNIQUE_SYMBOL PyArrayDaiCRF
#include <numpy/arrayobject.h>

using namespace dai;
using namespace std;
namespace bp = boost::python;

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

PyObject * mrf(PyArrayObject* unaries, PyArrayObject* edges, PyArrayObject* edge_potentials, string alg, size_t verbose) {
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
    if (verbose > 0)
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
                pairwise_factor.set(i + n_states * j, *((double*)PyArray_GETPTR2(edge_potentials, i, j)));
            }
        factors.push_back(pairwise_factor);
    }
    
    FactorGraph fg(factors);
    size_t maxiter = 100;
    Real   tol = 1e-9;
    vector<size_t> mpstate;

    // Store the constants in a PropertySet object
    PropertySet opts;
    if (alg == "jt")
    {
        JTree jt( fg, opts("updates",string("HUGIN"))("inference",string("MAXPROD")));
        jt.init();
        jt.run();
        mpstate = jt.findMaximum();
    }
    else if (alg == "treeep")
    {
        opts.set("type", string("ORG"));  // Maximum number of iterations
        opts.set("tol", 1e-4);          // Tolerance for convergence
        TreeEP ep(fg, opts);
        ep.init();
        ep.run();
        mpstate = ep.findMaximum();
    }
    else if (alg == "maxprod"){
        opts.set("maxiter", (size_t)10);  // Maximum number of iterations
        opts.set("tol", 1e-8);          // Tolerance for convergence
        opts.set("verbose",verbose);     // Verbosity (amount of output generated)

        BP mp(fg, opts("updates",string("SEQRND"))("logdomain",false)("inference",string("MAXPROD"))("damping",string("0.1")));
        mp.init();
        mp.run();
        mpstate = mp.findMaximum();
    }
    else if (alg == "gibbs"){
        opts.set("maxiter", size_t(100));   // number of Gibbs sampler iterations
        opts.set("burnin", size_t(0));
        opts.set("verbose", size_t(1));
        Gibbs gibbsSampler(fg, opts);
        gibbsSampler.init();
        gibbsSampler.run();
        mpstate = gibbsSampler.findMaximum();
    }
    else if (alg == "trw")
    {
        opts.set("tol", 1e-2);
        opts.set("logdomain", false);
        opts.set("updates", string("SEQRND"));
        TRWBP trw( fg, opts);
        trw.init();
        trw.run();
        mpstate = trw.findMaximum();
    }
    else {
        throw runtime_error("Invalid algorithm.");
    }
    npy_intp map_size = n_vertices;
    PyObject * map = PyArray_SimpleNew(1, &map_size, PyArray_INT);
    if (map == NULL)
        throw runtime_error("Could not allocate output array.");
    for(size_t i = 0; i < n_vertices; i++){
        ((int*)PyArray_GETPTR1(map, i))[0] = mpstate[i];
    }
    return map;
}


PyObject * potts_mrf(PyArrayObject* unaries, PyArrayObject* edges, double edge_strength, size_t verbose) {
    // validate input
    validate_unaries_edges(unaries, edges);
    npy_intp* unaries_dims = PyArray_DIMS(unaries);
    npy_intp* edges_dims = PyArray_DIMS(edges);
    int n_vertices = unaries_dims[0];
    int n_states = unaries_dims[1];
    int n_edges = edges_dims[0];
    if (verbose > 0)
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

    // Store the constants in a PropertySet object
    PropertySet opts;
    opts.set("maxiter",maxiter);  // Maximum number of iterations
    opts.set("tol",tol);          // Tolerance for convergence
    opts.set("verbose",verbose);     // Verbosity (amount of output generated)

    BP mp(fg, opts("updates",string("SEQRND"))("logdomain",false)("inference",string("MAXPROD"))("damping",string("0.1")));
    mp.init();
    mp.run();
    vector<size_t> mpstate = mp.findMaximum();
    
    npy_intp map_size = n_vertices;
    PyObject * map = PyArray_SimpleNew(1, &map_size, PyArray_INT);
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
    bp::converter::registry::insert(
	    &extract_pyarray, boost::python::type_id<PyArrayObject>());
    bp::def("potts_mrf", potts_mrf, (bp::arg("unaries"), bp::arg("egdges"), bp::arg("edge_strength"), bp::arg("verbose")=0));
    bp::def("mrf", mrf, (bp::arg("unaries"), bp::arg("egdges"), bp::arg("edge_weights"), bp::arg("alg"), bp::arg("verbose")=0));
    import_array();
}
