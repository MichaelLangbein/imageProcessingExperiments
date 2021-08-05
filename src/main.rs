extern crate nalgebra as na;
use na::{ DMatrix };


fn subtract_column_mean(data: &DMatrix<f32>) -> DMatrix<f32> {
    let column_mean = data.row_mean();
    
    let mut data_centered = data.clone();
    let (rows, cols) = data_centered.shape();
    for r in 0..rows {
        for c in 0..cols {
            data_centered[(r, c)] -= column_mean[c];
        } 
    }
    data_centered
}

fn get_covariance_matrix(data: &DMatrix<f32>) -> DMatrix<f32> {
    let (rows, _) = data.shape();
    let centered = subtract_column_mean(data);
    let cov_nonnorm = centered.transpose() * centered;
    let cov = cov_nonnorm.map(|val| val / (rows as f32));
    cov
}

fn pca_transform(data: &DMatrix<f32>) -> DMatrix<f32> {
    let cov = get_covariance_matrix(data);
    let eigen = cov.symmetric_eigen();
    let data_transformed = data * eigen.eigenvectors;
    data_transformed
}


fn main() {
    let data: DMatrix<f32> = DMatrix::from_row_slice(4, 3, &[
        2.2, 4.8, 2.1,
        6.4, 3.3, 7.3,
        8.1, 4.2, 8.1,
        1.2, 6.3, 3.1
    ]);
    // let data: OMatrix<f32, Dynamic, Dynamic> = OMatrix::new()
    let data_cov = get_covariance_matrix(&data);
    let data_transformed = pca_transform(&data);
    let data_transformed_cov = get_covariance_matrix(&data_transformed);

    println!("{}", data_cov);
    println!("{}", data_transformed_cov);
}