//! Small dense linear algebra (row-major), deterministic: covariance, ridge least squares, symmetric
//! eigen-decomposition (cyclic Jacobi). Sizes here are tiny (at most a few dozen).

/// Row-major matrix.
#[derive(Clone, Debug, PartialEq)]
pub struct Mat {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>,
}

impl Mat {
    pub fn zeros(rows: usize, cols: usize) -> Mat {
        Mat { rows, cols, data: vec![0.0; rows * cols] }
    }
    pub fn identity(n: usize) -> Mat {
        let mut m = Mat::zeros(n, n);
        for i in 0..n {
            m[(i, i)] = 1.0;
        }
        m
    }
    pub fn from_rows(rows: &[Vec<f64>]) -> Mat {
        let cols = rows.first().map_or(0, |r| r.len());
        let mut m = Mat::zeros(rows.len(), cols);
        for (i, r) in rows.iter().enumerate() {
            assert_eq!(r.len(), cols);
            m.data[i * cols..(i + 1) * cols].copy_from_slice(r);
        }
        m
    }
    pub fn t(&self) -> Mat {
        let mut m = Mat::zeros(self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                m[(j, i)] = self[(i, j)];
            }
        }
        m
    }
    pub fn mul(&self, o: &Mat) -> Mat {
        assert_eq!(self.cols, o.rows);
        let mut m = Mat::zeros(self.rows, o.cols);
        for i in 0..self.rows {
            for k in 0..self.cols {
                let a = self[(i, k)];
                for j in 0..o.cols {
                    m[(i, j)] += a * o[(k, j)];
                }
            }
        }
        m
    }
    pub fn mul_vec(&self, v: &[f64]) -> Vec<f64> {
        assert_eq!(self.cols, v.len());
        (0..self.rows).map(|i| (0..self.cols).map(|j| self[(i, j)] * v[j]).sum()).collect()
    }
}

impl std::ops::Index<(usize, usize)> for Mat {
    type Output = f64;
    fn index(&self, (i, j): (usize, usize)) -> &f64 {
        &self.data[i * self.cols + j]
    }
}
impl std::ops::IndexMut<(usize, usize)> for Mat {
    fn index_mut(&mut self, (i, j): (usize, usize)) -> &mut f64 {
        &mut self.data[i * self.cols + j]
    }
}

pub fn mean_rows(x: &[Vec<f64>]) -> Vec<f64> {
    let n = x.len() as f64;
    let d = x[0].len();
    let mut m = vec![0.0; d];
    for r in x {
        for j in 0..d {
            m[j] += r[j];
        }
    }
    m.iter_mut().for_each(|v| *v /= n);
    m
}

/// Covariance (1/n) of rows about their mean.
pub fn covariance(x: &[Vec<f64>]) -> Mat {
    let m = mean_rows(x);
    let d = m.len();
    let mut c = Mat::zeros(d, d);
    for r in x {
        for i in 0..d {
            for j in 0..d {
                c[(i, j)] += (r[i] - m[i]) * (r[j] - m[j]);
            }
        }
    }
    let n = x.len() as f64;
    c.data.iter_mut().for_each(|v| *v /= n);
    c
}

/// Solve the symmetric positive definite system A x = b for each column of B (Cholesky).
pub fn solve_spd(a: &Mat, b: &Mat) -> Mat {
    let n = a.rows;
    let mut l = Mat::zeros(n, n);
    for i in 0..n {
        for j in 0..=i {
            let mut s = a[(i, j)];
            for k in 0..j {
                s -= l[(i, k)] * l[(j, k)];
            }
            if i == j {
                assert!(s > 0.0, "matrix not positive definite");
                l[(i, i)] = s.sqrt();
            } else {
                l[(i, j)] = s / l[(j, j)];
            }
        }
    }
    let mut x = Mat::zeros(n, b.cols);
    for c in 0..b.cols {
        let mut y = vec![0.0; n];
        for i in 0..n {
            let mut s = b[(i, c)];
            for k in 0..i {
                s -= l[(i, k)] * y[k];
            }
            y[i] = s / l[(i, i)];
        }
        for i in (0..n).rev() {
            let mut s = y[i];
            for k in i + 1..n {
                s -= l[(k, i)] * x[(k, c)];
            }
            x[(i, c)] = s / l[(i, i)];
        }
    }
    x
}

/// Ridge least squares with an unpenalized intercept: returns (weights d_in x d_out, intercept d_out).
pub fn ridge(x: &[Vec<f64>], y: &[Vec<f64>], lambda: f64) -> (Mat, Vec<f64>) {
    let mx = mean_rows(x);
    let my = mean_rows(y);
    let (din, dout) = (mx.len(), my.len());
    let mut xtx = Mat::zeros(din, din);
    let mut xty = Mat::zeros(din, dout);
    for (rx, ry) in x.iter().zip(y) {
        for i in 0..din {
            let a = rx[i] - mx[i];
            for j in 0..din {
                xtx[(i, j)] += a * (rx[j] - mx[j]);
            }
            for j in 0..dout {
                xty[(i, j)] += a * (ry[j] - my[j]);
            }
        }
    }
    let n = x.len() as f64;
    for i in 0..din {
        xtx[(i, i)] += lambda * n;
    }
    let w = solve_spd(&xtx, &xty);
    let b: Vec<f64> = (0..dout).map(|j| my[j] - (0..din).map(|i| mx[i] * w[(i, j)]).sum::<f64>()).collect();
    (w, b)
}

pub fn ridge_apply(w: &Mat, b: &[f64], x: &[f64]) -> Vec<f64> {
    (0..w.cols).map(|j| b[j] + (0..w.rows).map(|i| x[i] * w[(i, j)]).sum::<f64>()).collect()
}

/// Symmetric eigen-decomposition by cyclic Jacobi rotations: (eigenvalues ascending, eigenvectors as columns).
pub fn sym_eigen(a: &Mat) -> (Vec<f64>, Mat) {
    let n = a.rows;
    let mut m = a.clone();
    let mut v = Mat::identity(n);
    for _sweep in 0..100 {
        let off: f64 = (0..n)
            .flat_map(|i| (0..n).filter(move |&j| j != i).map(move |j| (i, j)))
            .map(|(i, j)| m[(i, j)] * m[(i, j)])
            .sum();
        if off < 1e-24 {
            break;
        }
        for p in 0..n {
            for q in p + 1..n {
                if m[(p, q)].abs() < 1e-300 {
                    continue;
                }
                let theta = (m[(q, q)] - m[(p, p)]) / (2.0 * m[(p, q)]);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let t = if theta == 0.0 { 1.0 } else { t };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for k in 0..n {
                    let (mkp, mkq) = (m[(k, p)], m[(k, q)]);
                    m[(k, p)] = c * mkp - s * mkq;
                    m[(k, q)] = s * mkp + c * mkq;
                }
                for k in 0..n {
                    let (mpk, mqk) = (m[(p, k)], m[(q, k)]);
                    m[(p, k)] = c * mpk - s * mqk;
                    m[(q, k)] = s * mpk + c * mqk;
                }
                for k in 0..n {
                    let (vkp, vkq) = (v[(k, p)], v[(k, q)]);
                    v[(k, p)] = c * vkp - s * vkq;
                    v[(k, q)] = s * vkp + c * vkq;
                }
            }
        }
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| m[(i, i)].partial_cmp(&m[(j, j)]).unwrap().then(i.cmp(&j)));
    let values = order.iter().map(|&i| m[(i, i)]).collect();
    let mut vectors = Mat::zeros(n, n);
    for (c, &i) in order.iter().enumerate() {
        // Deterministic sign: the largest-magnitude component is positive.
        let col: Vec<f64> = (0..n).map(|k| v[(k, i)]).collect();
        let big = col.iter().cloned().fold(0.0f64, |a, x| if x.abs() > a.abs() { x } else { a });
        let sign = if big < 0.0 { -1.0 } else { 1.0 };
        for k in 0..n {
            vectors[(k, c)] = sign * col[k];
        }
    }
    (values, vectors)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eigen_reconstructs_and_ridge_recovers_a_linear_map() {
        let a = Mat::from_rows(&[vec![4.0, 1.0, 0.5], vec![1.0, 3.0, 0.2], vec![0.5, 0.2, 2.0]]);
        let (vals, vecs) = sym_eigen(&a);
        assert!(vals.windows(2).all(|w| w[0] <= w[1]));
        let mut d = Mat::zeros(3, 3);
        for i in 0..3 {
            d[(i, i)] = vals[i];
        }
        let back = vecs.mul(&d).mul(&vecs.t());
        for (x, y) in back.data.iter().zip(&a.data) {
            assert!((x - y).abs() < 1e-10);
        }
        let x: Vec<Vec<f64>> = (0..50).map(|i| vec![i as f64 * 0.1, ((i * 7) % 11) as f64]).collect();
        let y: Vec<Vec<f64>> = x.iter().map(|r| vec![2.0 * r[0] - r[1] + 3.0]).collect();
        let (w, b) = ridge(&x, &y, 0.0);
        assert!((w[(0, 0)] - 2.0).abs() < 1e-9 && (w[(1, 0)] + 1.0).abs() < 1e-9 && (b[0] - 3.0).abs() < 1e-9);
    }
}
