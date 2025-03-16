//
// Created by ianzh on 3/12/2025.
//

#ifndef CPUDATA_H
#define CPUDATA_H
#include "deviceData.h"
#include <vector>
#include <stdexcept>
class CPUData : public DeviceData {
    float *head;
    size_t size;
public:
    CPUData(size_t size);

    CPUData(const std::vector<float> &data_to_copy);

    CPUData(float *C, size_t size);

    CPUData(const DeviceData &other);

    CPUData &operator=(const DeviceData &other);

    CPUData(DeviceData &&other) noexcept;

    CPUData &operator=(DeviceData &&other) noexcept;

    ~CPUData();
    float *getData() const;
    size_t getSize() const;
    float *&getData();
    size_t &getSize();
    float iloc(int i) const;

    std::unique_ptr<DeviceData> elemAdd(const DeviceData *other) const;
    std::unique_ptr<DeviceData> elemSub(const DeviceData *other) const;
    std::unique_ptr<DeviceData> elemMult(const DeviceData *other) const;
    std::unique_ptr<DeviceData> mat_mult(const DeviceData *other, int m, int k, int n) const;

};

#endif //CPUDATA_H
