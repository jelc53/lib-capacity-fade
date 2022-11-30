// Julian Cooper 2022

functions {
    vector soh_decay(
        vector x,
        real alpha,
        real beta,
        real gamma
    ) {
        return 2 - exp(alpha*(pow(x, beta) - gamma))
    }
}
data {
    int<lower=0> T;             // total cycles observed
    int<lower=0> N;             // number of battery cells = 200
    int<lower=0> d;             // number of feature dimensions
    vector[N] y;
    matrix[N,d] X;
    vector[N] N_BC;              // cycle life for each battery cell
    
}
parameters {
    real alpha_0;
    real alpha_1;
    real alpha_2;
    real alpha_3;

    real beta_0;
    real beta_1;
    real beta_2;
    real beta_3;

    real gamma_0;
    real gamma_1;
    real gamma_2;
    real gamma_3;

    real<lower=0> sigma;
}
transformed parameters {
    vector[N]<lower=0> alpha;
    vector[N]<lower=0> beta;
    vector[N]<lower=0> gamma;
    vector[T] y_hat;
   {
    int idx = 1;
    real scaled_cycle_count; 

    for(i in 1:N) {
        alpha[i] = alpha_0 + alpha_1*X[i,1] + alpha_2*X[i,2] + alpha_3*X[i,3];
        beta[i] = beta_0 + beta_1*X[i,1] + beta_2*X[i,2] + beta_3*X[i,3];
        gamma[i] = gamma_0 + gamma_1*X[i,1] + gamma_2*X[i,2] + gamma_3*X[i,3];

        for (j in 1:N_BC[i]) {
            scaled_cycle_count = j / 1000;
            y_hat[idx] = soh_decay(scaled_cycle_count, alpha[i], beta[i], gamma[i]);
            idx += 1;
        }
    }
   }
}
model {
    alpha_0 ~ normal(0, 1);
    alpha_1 ~ normal(0, 1);
    alpha_2 ~ normal(0, 1);
    alpha_3 ~ normal(0, 1);

    beta_0 ~ normal(0, 1);
    beta_1 ~ normal(0, 1);
    beta_2 ~ normal(0, 1);
    beta_3 ~ normal(0, 1);

    gamma_0 ~ normal(0, 1);
    gamma_1 ~ normal(0, 1);
    gamma_2 ~ normal(0, 1);
    gamma_3 ~ normal(0, 1);

    sigma ~ gamma(1, 2);

    y ~ normal(y_hat, sigma)
}