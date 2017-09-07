#define RESTRICT __restrict__

OMP( declare target )
void star1(const int n, const int t, const double * RESTRICT in, double * RESTRICT out) {
    OMP_TARGET( teams distribute parallel for simd collapse(2) schedule(static,1) )
    for (auto i=1; i<n-1; ++i) {
      for (auto j=1; j<n-1; ++j) {
            out[i*n+j] += +in[(i+-1)*n+(j+0)] * -0.5
                          +in[(i+0)*n+(j+-1)] * -0.5
                          +in[(i+0)*n+(j+1)] * 0.5
                          +in[(i+1)*n+(j+0)] * 0.5;
       }
     }
}

void star2(const int n, const int t, const double * RESTRICT in, double * RESTRICT out) {
    OMP_TARGET( teams distribute parallel for simd collapse(2) schedule(static,1) )
    for (auto i=2; i<n-2; ++i) {
      for (auto j=2; j<n-2; ++j) {
            out[i*n+j] += +in[(i+-2)*n+(j+0)] * -0.125
                          +in[(i+-1)*n+(j+0)] * -0.25
                          +in[(i+0)*n+(j+-2)] * -0.125
                          +in[(i+0)*n+(j+-1)] * -0.25
                          +in[(i+0)*n+(j+1)] * 0.25
                          +in[(i+0)*n+(j+2)] * 0.125
                          +in[(i+1)*n+(j+0)] * 0.25
                          +in[(i+2)*n+(j+0)] * 0.125;
       }
     }
}

void star3(const int n, const int t, const double * RESTRICT in, double * RESTRICT out) {
    OMP_TARGET( teams distribute parallel for simd collapse(2) schedule(static,1) )
    for (auto i=3; i<n-3; ++i) {
      for (auto j=3; j<n-3; ++j) {
            out[i*n+j] += +in[(i+-3)*n+(j+0)] * -0.05555555555555555
                          +in[(i+-2)*n+(j+0)] * -0.08333333333333333
                          +in[(i+-1)*n+(j+0)] * -0.16666666666666666
                          +in[(i+0)*n+(j+-3)] * -0.05555555555555555
                          +in[(i+0)*n+(j+-2)] * -0.08333333333333333
                          +in[(i+0)*n+(j+-1)] * -0.16666666666666666
                          +in[(i+0)*n+(j+1)] * 0.16666666666666666
                          +in[(i+0)*n+(j+2)] * 0.08333333333333333
                          +in[(i+0)*n+(j+3)] * 0.05555555555555555
                          +in[(i+1)*n+(j+0)] * 0.16666666666666666
                          +in[(i+2)*n+(j+0)] * 0.08333333333333333
                          +in[(i+3)*n+(j+0)] * 0.05555555555555555;
       }
     }
}

void star4(const int n, const int t, const double * RESTRICT in, double * RESTRICT out) {
    OMP_TARGET( teams distribute parallel for simd collapse(2) schedule(static,1) )
    for (auto i=4; i<n-4; ++i) {
      for (auto j=4; j<n-4; ++j) {
            out[i*n+j] += +in[(i+-4)*n+(j+0)] * -0.03125
                          +in[(i+-3)*n+(j+0)] * -0.041666666666666664
                          +in[(i+-2)*n+(j+0)] * -0.0625
                          +in[(i+-1)*n+(j+0)] * -0.125
                          +in[(i+0)*n+(j+-4)] * -0.03125
                          +in[(i+0)*n+(j+-3)] * -0.041666666666666664
                          +in[(i+0)*n+(j+-2)] * -0.0625
                          +in[(i+0)*n+(j+-1)] * -0.125
                          +in[(i+0)*n+(j+1)] * 0.125
                          +in[(i+0)*n+(j+2)] * 0.0625
                          +in[(i+0)*n+(j+3)] * 0.041666666666666664
                          +in[(i+0)*n+(j+4)] * 0.03125
                          +in[(i+1)*n+(j+0)] * 0.125
                          +in[(i+2)*n+(j+0)] * 0.0625
                          +in[(i+3)*n+(j+0)] * 0.041666666666666664
                          +in[(i+4)*n+(j+0)] * 0.03125;
       }
     }
}

void star5(const int n, const int t, const double * RESTRICT in, double * RESTRICT out) {
    OMP_TARGET( teams distribute parallel for simd collapse(2) schedule(static,1) )
    for (auto i=5; i<n-5; ++i) {
      for (auto j=5; j<n-5; ++j) {
            out[i*n+j] += +in[(i+-5)*n+(j+0)] * -0.02
                          +in[(i+-4)*n+(j+0)] * -0.025
                          +in[(i+-3)*n+(j+0)] * -0.03333333333333333
                          +in[(i+-2)*n+(j+0)] * -0.05
                          +in[(i+-1)*n+(j+0)] * -0.1
                          +in[(i+0)*n+(j+-5)] * -0.02
                          +in[(i+0)*n+(j+-4)] * -0.025
                          +in[(i+0)*n+(j+-3)] * -0.03333333333333333
                          +in[(i+0)*n+(j+-2)] * -0.05
                          +in[(i+0)*n+(j+-1)] * -0.1
                          +in[(i+0)*n+(j+1)] * 0.1
                          +in[(i+0)*n+(j+2)] * 0.05
                          +in[(i+0)*n+(j+3)] * 0.03333333333333333
                          +in[(i+0)*n+(j+4)] * 0.025
                          +in[(i+0)*n+(j+5)] * 0.02
                          +in[(i+1)*n+(j+0)] * 0.1
                          +in[(i+2)*n+(j+0)] * 0.05
                          +in[(i+3)*n+(j+0)] * 0.03333333333333333
                          +in[(i+4)*n+(j+0)] * 0.025
                          +in[(i+5)*n+(j+0)] * 0.02;
       }
     }
}

void grid1(const int n, const int t, const double * RESTRICT in, double * RESTRICT out) {
    OMP_TARGET( teams distribute parallel for simd collapse(2) schedule(static,1) )
    for (auto i=1; i<n-1; ++i) {
      for (auto j=1; j<n-1; ++j) {
            out[i*n+j] += +in[(i+-1)*n+(j+-1)] * -0.25
                          +in[(i+-1)*n+(j+0)] * -0.25
                          +in[(i+0)*n+(j+-1)] * -0.25
                          +in[(i+0)*n+(j+1)] * 0.25
                          +in[(i+1)*n+(j+0)] * 0.25
                          +in[(i+1)*n+(j+1)] * 0.25
                          ;
       }
     }
}

void grid2(const int n, const int t, const double * RESTRICT in, double * RESTRICT out) {
    OMP_TARGET( teams distribute parallel for simd collapse(2) schedule(static,1) )
    for (auto i=2; i<n-2; ++i) {
      for (auto j=2; j<n-2; ++j) {
            out[i*n+j] += +in[(i+-2)*n+(j+-2)] * -0.0625
                          +in[(i+-2)*n+(j+-1)] * -0.020833333333333332
                          +in[(i+-2)*n+(j+0)] * -0.020833333333333332
                          +in[(i+-2)*n+(j+1)] * -0.020833333333333332
                          +in[(i+-1)*n+(j+-2)] * -0.020833333333333332
                          +in[(i+-1)*n+(j+-1)] * -0.125
                          +in[(i+-1)*n+(j+0)] * -0.125
                          +in[(i+-1)*n+(j+2)] * 0.020833333333333332
                          +in[(i+0)*n+(j+-2)] * -0.020833333333333332
                          +in[(i+0)*n+(j+-1)] * -0.125
                          +in[(i+0)*n+(j+1)] * 0.125
                          +in[(i+0)*n+(j+2)] * 0.020833333333333332
                          +in[(i+1)*n+(j+-2)] * -0.020833333333333332
                          +in[(i+1)*n+(j+0)] * 0.125
                          +in[(i+1)*n+(j+1)] * 0.125
                          +in[(i+1)*n+(j+2)] * 0.020833333333333332
                          +in[(i+2)*n+(j+-1)] * 0.020833333333333332
                          +in[(i+2)*n+(j+0)] * 0.020833333333333332
                          +in[(i+2)*n+(j+1)] * 0.020833333333333332
                          +in[(i+2)*n+(j+2)] * 0.0625
                          ;
       }
     }
}

void grid3(const int n, const int t, const double * RESTRICT in, double * RESTRICT out) {
    OMP_TARGET( teams distribute parallel for simd collapse(2) schedule(static,1) )
    for (auto i=3; i<n-3; ++i) {
      for (auto j=3; j<n-3; ++j) {
            out[i*n+j] += +in[(i+-3)*n+(j+-3)] * -0.027777777777777776
                          +in[(i+-3)*n+(j+-2)] * -0.005555555555555556
                          +in[(i+-3)*n+(j+-1)] * -0.005555555555555556
                          +in[(i+-3)*n+(j+0)] * -0.005555555555555556
                          +in[(i+-3)*n+(j+1)] * -0.005555555555555556
                          +in[(i+-3)*n+(j+2)] * -0.005555555555555556
                          +in[(i+-2)*n+(j+-3)] * -0.005555555555555556
                          +in[(i+-2)*n+(j+-2)] * -0.041666666666666664
                          +in[(i+-2)*n+(j+-1)] * -0.013888888888888888
                          +in[(i+-2)*n+(j+0)] * -0.013888888888888888
                          +in[(i+-2)*n+(j+1)] * -0.013888888888888888
                          +in[(i+-2)*n+(j+3)] * 0.005555555555555556
                          +in[(i+-1)*n+(j+-3)] * -0.005555555555555556
                          +in[(i+-1)*n+(j+-2)] * -0.013888888888888888
                          +in[(i+-1)*n+(j+-1)] * -0.08333333333333333
                          +in[(i+-1)*n+(j+0)] * -0.08333333333333333
                          +in[(i+-1)*n+(j+2)] * 0.013888888888888888
                          +in[(i+-1)*n+(j+3)] * 0.005555555555555556
                          +in[(i+0)*n+(j+-3)] * -0.005555555555555556
                          +in[(i+0)*n+(j+-2)] * -0.013888888888888888
                          +in[(i+0)*n+(j+-1)] * -0.08333333333333333
                          +in[(i+0)*n+(j+1)] * 0.08333333333333333
                          +in[(i+0)*n+(j+2)] * 0.013888888888888888
                          +in[(i+0)*n+(j+3)] * 0.005555555555555556
                          +in[(i+1)*n+(j+-3)] * -0.005555555555555556
                          +in[(i+1)*n+(j+-2)] * -0.013888888888888888
                          +in[(i+1)*n+(j+0)] * 0.08333333333333333
                          +in[(i+1)*n+(j+1)] * 0.08333333333333333
                          +in[(i+1)*n+(j+2)] * 0.013888888888888888
                          +in[(i+1)*n+(j+3)] * 0.005555555555555556
                          +in[(i+2)*n+(j+-3)] * -0.005555555555555556
                          +in[(i+2)*n+(j+-1)] * 0.013888888888888888
                          +in[(i+2)*n+(j+0)] * 0.013888888888888888
                          +in[(i+2)*n+(j+1)] * 0.013888888888888888
                          +in[(i+2)*n+(j+2)] * 0.041666666666666664
                          +in[(i+2)*n+(j+3)] * 0.005555555555555556
                          +in[(i+3)*n+(j+-2)] * 0.005555555555555556
                          +in[(i+3)*n+(j+-1)] * 0.005555555555555556
                          +in[(i+3)*n+(j+0)] * 0.005555555555555556
                          +in[(i+3)*n+(j+1)] * 0.005555555555555556
                          +in[(i+3)*n+(j+2)] * 0.005555555555555556
                          +in[(i+3)*n+(j+3)] * 0.027777777777777776
                          ;
       }
     }
}

void grid4(const int n, const int t, const double * RESTRICT in, double * RESTRICT out) {
    OMP_TARGET( teams distribute parallel for simd collapse(2) schedule(static,1) )
    for (auto i=4; i<n-4; ++i) {
      for (auto j=4; j<n-4; ++j) {
            out[i*n+j] += +in[(i+-4)*n+(j+-4)] * -0.015625
                          +in[(i+-4)*n+(j+-3)] * -0.002232142857142857
                          +in[(i+-4)*n+(j+-2)] * -0.002232142857142857
                          +in[(i+-4)*n+(j+-1)] * -0.002232142857142857
                          +in[(i+-4)*n+(j+0)] * -0.002232142857142857
                          +in[(i+-4)*n+(j+1)] * -0.002232142857142857
                          +in[(i+-4)*n+(j+2)] * -0.002232142857142857
                          +in[(i+-4)*n+(j+3)] * -0.002232142857142857
                          +in[(i+-3)*n+(j+-4)] * -0.002232142857142857
                          +in[(i+-3)*n+(j+-3)] * -0.020833333333333332
                          +in[(i+-3)*n+(j+-2)] * -0.004166666666666667
                          +in[(i+-3)*n+(j+-1)] * -0.004166666666666667
                          +in[(i+-3)*n+(j+0)] * -0.004166666666666667
                          +in[(i+-3)*n+(j+1)] * -0.004166666666666667
                          +in[(i+-3)*n+(j+2)] * -0.004166666666666667
                          +in[(i+-3)*n+(j+4)] * 0.002232142857142857
                          +in[(i+-2)*n+(j+-4)] * -0.002232142857142857
                          +in[(i+-2)*n+(j+-3)] * -0.004166666666666667
                          +in[(i+-2)*n+(j+-2)] * -0.03125
                          +in[(i+-2)*n+(j+-1)] * -0.010416666666666666
                          +in[(i+-2)*n+(j+0)] * -0.010416666666666666
                          +in[(i+-2)*n+(j+1)] * -0.010416666666666666
                          +in[(i+-2)*n+(j+3)] * 0.004166666666666667
                          +in[(i+-2)*n+(j+4)] * 0.002232142857142857
                          +in[(i+-1)*n+(j+-4)] * -0.002232142857142857
                          +in[(i+-1)*n+(j+-3)] * -0.004166666666666667
                          +in[(i+-1)*n+(j+-2)] * -0.010416666666666666
                          +in[(i+-1)*n+(j+-1)] * -0.0625
                          +in[(i+-1)*n+(j+0)] * -0.0625
                          +in[(i+-1)*n+(j+2)] * 0.010416666666666666
                          +in[(i+-1)*n+(j+3)] * 0.004166666666666667
                          +in[(i+-1)*n+(j+4)] * 0.002232142857142857
                          +in[(i+0)*n+(j+-4)] * -0.002232142857142857
                          +in[(i+0)*n+(j+-3)] * -0.004166666666666667
                          +in[(i+0)*n+(j+-2)] * -0.010416666666666666
                          +in[(i+0)*n+(j+-1)] * -0.0625
                          +in[(i+0)*n+(j+1)] * 0.0625
                          +in[(i+0)*n+(j+2)] * 0.010416666666666666
                          +in[(i+0)*n+(j+3)] * 0.004166666666666667
                          +in[(i+0)*n+(j+4)] * 0.002232142857142857
                          +in[(i+1)*n+(j+-4)] * -0.002232142857142857
                          +in[(i+1)*n+(j+-3)] * -0.004166666666666667
                          +in[(i+1)*n+(j+-2)] * -0.010416666666666666
                          +in[(i+1)*n+(j+0)] * 0.0625
                          +in[(i+1)*n+(j+1)] * 0.0625
                          +in[(i+1)*n+(j+2)] * 0.010416666666666666
                          +in[(i+1)*n+(j+3)] * 0.004166666666666667
                          +in[(i+1)*n+(j+4)] * 0.002232142857142857
                          +in[(i+2)*n+(j+-4)] * -0.002232142857142857
                          +in[(i+2)*n+(j+-3)] * -0.004166666666666667
                          +in[(i+2)*n+(j+-1)] * 0.010416666666666666
                          +in[(i+2)*n+(j+0)] * 0.010416666666666666
                          +in[(i+2)*n+(j+1)] * 0.010416666666666666
                          +in[(i+2)*n+(j+2)] * 0.03125
                          +in[(i+2)*n+(j+3)] * 0.004166666666666667
                          +in[(i+2)*n+(j+4)] * 0.002232142857142857
                          +in[(i+3)*n+(j+-4)] * -0.002232142857142857
                          +in[(i+3)*n+(j+-2)] * 0.004166666666666667
                          +in[(i+3)*n+(j+-1)] * 0.004166666666666667
                          +in[(i+3)*n+(j+0)] * 0.004166666666666667
                          +in[(i+3)*n+(j+1)] * 0.004166666666666667
                          +in[(i+3)*n+(j+2)] * 0.004166666666666667
                          +in[(i+3)*n+(j+3)] * 0.020833333333333332
                          +in[(i+3)*n+(j+4)] * 0.002232142857142857
                          +in[(i+4)*n+(j+-3)] * 0.002232142857142857
                          +in[(i+4)*n+(j+-2)] * 0.002232142857142857
                          +in[(i+4)*n+(j+-1)] * 0.002232142857142857
                          +in[(i+4)*n+(j+0)] * 0.002232142857142857
                          +in[(i+4)*n+(j+1)] * 0.002232142857142857
                          +in[(i+4)*n+(j+2)] * 0.002232142857142857
                          +in[(i+4)*n+(j+3)] * 0.002232142857142857
                          +in[(i+4)*n+(j+4)] * 0.015625
                          ;
       }
     }
}

void grid5(const int n, const int t, const double * RESTRICT in, double * RESTRICT out) {
    OMP_TARGET( teams distribute parallel for simd collapse(2) schedule(static,1) )
    for (auto i=5; i<n-5; ++i) {
      for (auto j=5; j<n-5; ++j) {
            out[i*n+j] += +in[(i+-5)*n+(j+-5)] * -0.01
                          +in[(i+-5)*n+(j+-4)] * -0.0011111111111111111
                          +in[(i+-5)*n+(j+-3)] * -0.0011111111111111111
                          +in[(i+-5)*n+(j+-2)] * -0.0011111111111111111
                          +in[(i+-5)*n+(j+-1)] * -0.0011111111111111111
                          +in[(i+-5)*n+(j+0)] * -0.0011111111111111111
                          +in[(i+-5)*n+(j+1)] * -0.0011111111111111111
                          +in[(i+-5)*n+(j+2)] * -0.0011111111111111111
                          +in[(i+-5)*n+(j+3)] * -0.0011111111111111111
                          +in[(i+-5)*n+(j+4)] * -0.0011111111111111111
                          +in[(i+-4)*n+(j+-5)] * -0.0011111111111111111
                          +in[(i+-4)*n+(j+-4)] * -0.0125
                          +in[(i+-4)*n+(j+-3)] * -0.0017857142857142857
                          +in[(i+-4)*n+(j+-2)] * -0.0017857142857142857
                          +in[(i+-4)*n+(j+-1)] * -0.0017857142857142857
                          +in[(i+-4)*n+(j+0)] * -0.0017857142857142857
                          +in[(i+-4)*n+(j+1)] * -0.0017857142857142857
                          +in[(i+-4)*n+(j+2)] * -0.0017857142857142857
                          +in[(i+-4)*n+(j+3)] * -0.0017857142857142857
                          +in[(i+-4)*n+(j+5)] * 0.0011111111111111111
                          +in[(i+-3)*n+(j+-5)] * -0.0011111111111111111
                          +in[(i+-3)*n+(j+-4)] * -0.0017857142857142857
                          +in[(i+-3)*n+(j+-3)] * -0.016666666666666666
                          +in[(i+-3)*n+(j+-2)] * -0.0033333333333333335
                          +in[(i+-3)*n+(j+-1)] * -0.0033333333333333335
                          +in[(i+-3)*n+(j+0)] * -0.0033333333333333335
                          +in[(i+-3)*n+(j+1)] * -0.0033333333333333335
                          +in[(i+-3)*n+(j+2)] * -0.0033333333333333335
                          +in[(i+-3)*n+(j+4)] * 0.0017857142857142857
                          +in[(i+-3)*n+(j+5)] * 0.0011111111111111111
                          +in[(i+-2)*n+(j+-5)] * -0.0011111111111111111
                          +in[(i+-2)*n+(j+-4)] * -0.0017857142857142857
                          +in[(i+-2)*n+(j+-3)] * -0.0033333333333333335
                          +in[(i+-2)*n+(j+-2)] * -0.025
                          +in[(i+-2)*n+(j+-1)] * -0.008333333333333333
                          +in[(i+-2)*n+(j+0)] * -0.008333333333333333
                          +in[(i+-2)*n+(j+1)] * -0.008333333333333333
                          +in[(i+-2)*n+(j+3)] * 0.0033333333333333335
                          +in[(i+-2)*n+(j+4)] * 0.0017857142857142857
                          +in[(i+-2)*n+(j+5)] * 0.0011111111111111111
                          +in[(i+-1)*n+(j+-5)] * -0.0011111111111111111
                          +in[(i+-1)*n+(j+-4)] * -0.0017857142857142857
                          +in[(i+-1)*n+(j+-3)] * -0.0033333333333333335
                          +in[(i+-1)*n+(j+-2)] * -0.008333333333333333
                          +in[(i+-1)*n+(j+-1)] * -0.05
                          +in[(i+-1)*n+(j+0)] * -0.05
                          +in[(i+-1)*n+(j+2)] * 0.008333333333333333
                          +in[(i+-1)*n+(j+3)] * 0.0033333333333333335
                          +in[(i+-1)*n+(j+4)] * 0.0017857142857142857
                          +in[(i+-1)*n+(j+5)] * 0.0011111111111111111
                          +in[(i+0)*n+(j+-5)] * -0.0011111111111111111
                          +in[(i+0)*n+(j+-4)] * -0.0017857142857142857
                          +in[(i+0)*n+(j+-3)] * -0.0033333333333333335
                          +in[(i+0)*n+(j+-2)] * -0.008333333333333333
                          +in[(i+0)*n+(j+-1)] * -0.05
                          +in[(i+0)*n+(j+1)] * 0.05
                          +in[(i+0)*n+(j+2)] * 0.008333333333333333
                          +in[(i+0)*n+(j+3)] * 0.0033333333333333335
                          +in[(i+0)*n+(j+4)] * 0.0017857142857142857
                          +in[(i+0)*n+(j+5)] * 0.0011111111111111111
                          +in[(i+1)*n+(j+-5)] * -0.0011111111111111111
                          +in[(i+1)*n+(j+-4)] * -0.0017857142857142857
                          +in[(i+1)*n+(j+-3)] * -0.0033333333333333335
                          +in[(i+1)*n+(j+-2)] * -0.008333333333333333
                          +in[(i+1)*n+(j+0)] * 0.05
                          +in[(i+1)*n+(j+1)] * 0.05
                          +in[(i+1)*n+(j+2)] * 0.008333333333333333
                          +in[(i+1)*n+(j+3)] * 0.0033333333333333335
                          +in[(i+1)*n+(j+4)] * 0.0017857142857142857
                          +in[(i+1)*n+(j+5)] * 0.0011111111111111111
                          +in[(i+2)*n+(j+-5)] * -0.0011111111111111111
                          +in[(i+2)*n+(j+-4)] * -0.0017857142857142857
                          +in[(i+2)*n+(j+-3)] * -0.0033333333333333335
                          +in[(i+2)*n+(j+-1)] * 0.008333333333333333
                          +in[(i+2)*n+(j+0)] * 0.008333333333333333
                          +in[(i+2)*n+(j+1)] * 0.008333333333333333
                          +in[(i+2)*n+(j+2)] * 0.025
                          +in[(i+2)*n+(j+3)] * 0.0033333333333333335
                          +in[(i+2)*n+(j+4)] * 0.0017857142857142857
                          +in[(i+2)*n+(j+5)] * 0.0011111111111111111
                          +in[(i+3)*n+(j+-5)] * -0.0011111111111111111
                          +in[(i+3)*n+(j+-4)] * -0.0017857142857142857
                          +in[(i+3)*n+(j+-2)] * 0.0033333333333333335
                          +in[(i+3)*n+(j+-1)] * 0.0033333333333333335
                          +in[(i+3)*n+(j+0)] * 0.0033333333333333335
                          +in[(i+3)*n+(j+1)] * 0.0033333333333333335
                          +in[(i+3)*n+(j+2)] * 0.0033333333333333335
                          +in[(i+3)*n+(j+3)] * 0.016666666666666666
                          +in[(i+3)*n+(j+4)] * 0.0017857142857142857
                          +in[(i+3)*n+(j+5)] * 0.0011111111111111111
                          +in[(i+4)*n+(j+-5)] * -0.0011111111111111111
                          +in[(i+4)*n+(j+-3)] * 0.0017857142857142857
                          +in[(i+4)*n+(j+-2)] * 0.0017857142857142857
                          +in[(i+4)*n+(j+-1)] * 0.0017857142857142857
                          +in[(i+4)*n+(j+0)] * 0.0017857142857142857
                          +in[(i+4)*n+(j+1)] * 0.0017857142857142857
                          +in[(i+4)*n+(j+2)] * 0.0017857142857142857
                          +in[(i+4)*n+(j+3)] * 0.0017857142857142857
                          +in[(i+4)*n+(j+4)] * 0.0125
                          +in[(i+4)*n+(j+5)] * 0.0011111111111111111
                          +in[(i+5)*n+(j+-4)] * 0.0011111111111111111
                          +in[(i+5)*n+(j+-3)] * 0.0011111111111111111
                          +in[(i+5)*n+(j+-2)] * 0.0011111111111111111
                          +in[(i+5)*n+(j+-1)] * 0.0011111111111111111
                          +in[(i+5)*n+(j+0)] * 0.0011111111111111111
                          +in[(i+5)*n+(j+1)] * 0.0011111111111111111
                          +in[(i+5)*n+(j+2)] * 0.0011111111111111111
                          +in[(i+5)*n+(j+3)] * 0.0011111111111111111
                          +in[(i+5)*n+(j+4)] * 0.0011111111111111111
                          +in[(i+5)*n+(j+5)] * 0.01
                          ;
       }
     }
}

OMP( end declare target )
