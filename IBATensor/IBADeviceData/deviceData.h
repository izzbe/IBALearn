//
// Created by ianzh on 3/12/2025.
//

#ifndef DEVICEDATA_H
#define DEVICEDATA_H
#include <memory>

class DeviceData {

public:
    virtual ~DeviceData() = default;
    virtual float *getData() const = 0;
    virtual size_t getSize() const = 0;
    virtual float *&getData() = 0;
    virtual size_t &getSize() = 0;
    virtual float iloc(int i) const = 0;

    virtual std::unique_ptr<DeviceData> elemAdd(const DeviceData *other) const = 0;
    virtual std::unique_ptr<DeviceData> elemSub(const DeviceData *other) const = 0;
    virtual std::unique_ptr<DeviceData> elemMult(const DeviceData *other) const = 0;
    virtual std::unique_ptr<DeviceData> mat_mult(const DeviceData *other, int m, int k, int n) const = 0;

};

#endif //DEVICEDATA_H
