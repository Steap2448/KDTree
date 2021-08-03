#include "kd-tree.h"

double SurfaceAreaHeuristic::operator()(double SAL, double SAR, double SAP, int NL, int NR) const {
    return (ct_ + ci_ * ((SAL * NL + SAR * NR) / (SAP)));
}
