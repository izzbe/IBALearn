//
// Created by ianzh on 3/12/2025.
//
#include "CPUData.h"
#include <cstring>

// ------------------------------------------------ Constructors ------------------------------------------------------
CPUData::CPUData(size_t size) : head(new float[size]), size(size) {}

CPUData::CPUData(const std::vector<float> &data_to_copy) : CPUData(data_to_copy.size()) {
    int i = 0;
    for (auto it = data_to_copy.begin(); it != data_to_copy.end(); ++it) {
        head[i] = *it;
        ++i;
    }
}

CPUData::CPUData(float *C, size_t size) : head(C), size(size) {}

// -------------------------------------------------- Big 5 ----------------------------------------------------------
CPUData::CPUData(const DeviceData &other) : head(new float[other.getSize()]), size(other.getSize()) {
    memcpy(head, other.getData(), size * sizeof(float));
}

CPUData &CPUData::operator=(const DeviceData &other) {

    if (this == &other) { return *this; };

    CPUData temp(other);

    std::swap(head, temp.head);
    std::swap(size, temp.size);

    return *this;
}

CPUData::CPUData(DeviceData &&other) noexcept : head(other.getData()), size(other.getSize()) {
    other.getSize() = 0;
    other.getData() = nullptr;
}

CPUData &CPUData::operator=(DeviceData &&other) noexcept {
    if (this == &other) { return *this; }
    CPUData Temp(head, size);
    head = other.getData();
    size = other.getSize();

    other.getSize() = 0;
    other.getData() = nullptr;
    return *this;
}

CPUData::~CPUData() {
    delete head;
}

// -------------------------------------------------- Getters --------------------------------------------------------
float *CPUData::getData() const {
    return head;
}

size_t CPUData::getSize() const {
    return size;
}

float *&CPUData::getData() {
    return head;
}

size_t &CPUData::getSize() {
    return size;
}

float CPUData::iloc(int i) const {
    if (i < 0 || i >= size) {
        throw std::logic_error("Iloc index out of bounds");
    }
    return head[i];
}

void CPUData::set_index(int i, float val) {
    if(i < 0 || i > size) {
        throw std::logic_error("CUDA set_index out of bound");
    }

    head[i] = val;
}
// ------------------------------------------------- Matrix Ops -------------------------------------------------------
std::unique_ptr<DeviceData> CPUData::elemAdd(const DeviceData *other, int B_grouping, int B_stride) const {
    float *C = new float[size];
    for (int i = 0; i < size; i++) {
        C[i] = head[i] + other->iloc(i % B_grouping);
    }
    return std::make_unique<CPUData>(C, size);
}

std::unique_ptr<DeviceData> CPUData::elemSub(const DeviceData *other, int B_grouping, int B_stride) const {
    float *C = new float[size];
    for (int i = 0; i < size; i++) {
        C[i] = head[i] - other->iloc(i % B_grouping);
    }
    return std::make_unique<CPUData>(C, size);
}
std::unique_ptr<DeviceData> CPUData::elemMult(const DeviceData *other, int B_grouping, int B_stride) const {
    float *C = new float[size];
    for (int i = 0; i < size; i++) {
        C[i] = head[i] * other->iloc(i % B_grouping);
    }
    return std::make_unique<CPUData>(C, size);
}

std::unique_ptr<DeviceData> CPUData::mat_mult(const DeviceData *other, int H, int shared_axis, int W, int N) const {
    return std::make_unique<CPUData> (nullptr, 0);

}

std::unique_ptr<DeviceData> CPUData::conv2d(const DeviceData *kern, int N, int C_in,
                                             int H, int W, int H_out, int W_out, int K, int P, int S, int C_out) const {
    return std::make_unique<CPUData> (nullptr, 0);

}

std::unique_ptr<DeviceData> CPUData::avg_pool(int N, int C_in, int H, int W, int H_out, int W_out, int K, int P, int S) const {
    return std::make_unique<CPUData> (nullptr, 0);
}

std::unique_ptr<DeviceData> CPUData::max_pool(int N, int C_in, int H, int W, int H_out, int W_out, int K, int P, int S) const {
    return std::make_unique<CPUData> (nullptr, 0);
}

std::unique_ptr<DeviceData> CPUData::mat_transpose(int H, int W, int C, int N) const {
    return std::make_unique<CPUData> (nullptr, 0);
}

std::unique_ptr<DeviceData> CPUData::relu(int H, int W, int C, int N) const {
    return std::make_unique<CPUData> (nullptr, 0);
}

// -------------------------------------------------------- backwards -------------------------------------------------------

std::unique_ptr<DeviceData> CPUData::conv2d_backward_wr_kernel(const DeviceData *sigma,
                                            int C_k, int K, int H_in, int W_in, int C_in, int C_sigma, int H_sigma, int W_sigma, int P, int S, int N) const {
    return std::make_unique<CPUData> (nullptr, 0);
}

std::unique_ptr<DeviceData> CPUData::conv2d_backward_wr_input(const DeviceData *sigma, const DeviceData *kernel, int H_in, int W_in, int K, int C_in_k, int C_out_k,
                                                                 int sigma_H, int sigma_W, int sigma_C, int sigma_N, int P, int S) const {
    return std::make_unique<CPUData> (nullptr, 0);
}

