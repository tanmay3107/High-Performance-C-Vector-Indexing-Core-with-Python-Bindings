#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <queue>
#include <cmath>
#include <stdexcept>

namespace py = pybind11;

class VectorIndexCore {
private:
    size_t dim;
    std::vector<std::vector<float>> vectors;
    std::vector<int64_t> ids;

    // Euclidean distance calculation
    static float compute_distance(const float* a, const float* b, size_t d) {
        float sum = 0.0f;
        for (size_t i = 0; i < d; ++i) {
            float diff = a[i] - b[i];
            sum += diff * diff;
        }
        return std::sqrt(sum);
    }

public:
    VectorIndexCore(size_t dimension) : dim(dimension) {}

    void add_vectors(py::array_t<float> input_vectors, py::array_t<int64_t> input_ids) {
        auto v_buf = input_vectors.request();
        auto id_buf = input_ids.request();

        if (v_buf.ndim != 2) throw std::runtime_error("Vectors must be a 2D array.");
        if (v_buf.shape[1] != static_cast<ssize_t>(dim)) throw std::runtime_error("Dimension mismatch.");
        if (v_buf.shape[0] != id_buf.shape[0]) throw std::runtime_error("Number of vectors and IDs must match.");

        const float* v_ptr = static_cast<float*>(v_buf.ptr);
        const int64_t* id_ptr = static_cast<int64_t*>(id_buf.ptr);

        size_t num_rows = v_buf.shape[0];
        for (size_t i = 0; i < num_rows; ++i) {
            std::vector<float> vec(v_ptr + i * dim, v_ptr + (i + 1) * dim);
            vectors.push_back(vec);
            ids.push_back(id_ptr[i]);
        }
    }

    py::tuple search(py::array_t<float> query_vector, size_t k) {
        auto q_buf = query_vector.request();
        if (q_buf.size != static_cast<ssize_t>(dim)) throw std::runtime_error("Query dimension mismatch.");

        const float* q_ptr = static_cast<float*>(q_buf.ptr);

        // Min-priority queue to keep track of top-k closest elements
        auto comp = [](const std::pair<float, int64_t>& a, const std::pair<float, int64_t>& b) {
            return a.first < b.first; // Max-heap based on distance to extract top-k smallest
        };
        std::priority_queue<std::pair<float, int64_t>, std::vector<std::pair<float, int64_t>>, decltype(comp)> max_heap(comp);

        for (size_t i = 0; i < vectors.size(); ++i) {
            float dist = compute_distance(q_ptr, vectors[i].data(), dim);
            if (max_heap.size() < k) {
                max_heap.push({dist, ids[i]});
            } else if (dist < max_heap.top().first) {
                max_heap.pop();
                max_heap.push({dist, ids[i]});
            }
        }

        size_t results_size = max_heap.size();
        py::array_t<int64_t> result_ids(results_size);
        py::array_t<float> result_distances(results_size);

        auto id_p = result_ids.mutable_unchecked<1>();
        auto dist_p = result_distances.mutable_unchecked<1>();

        for (int i = results_size - 1; i >= 0; --i) {
            auto top = max_heap.top();
            max_heap.pop();
            id_p(i) = top.second;
            dist_p(i) = top.first;
        }

        return py::make_tuple(result_ids, result_distances);
    }

    size_t size() const {
        return vectors.size();
    }
};

// Expose module using pybind11
PYBIND11_MODULE(vector_core, m) {
    m.doc = "High-performance C++ Vector Indexing Core";
    py::class_<VectorIndexCore>(m, "VectorIndexCore")
        .def(py::init<size_t>(), py::arg("dimension"))
        .def("add_vectors", &VectorIndexCore::add_vectors, "Add batch of vectors and IDs")
        .def("search", &VectorIndexCore::search, "Perform exact k-NN search")
        .def("__len__", &VectorIndexCore::size);
}