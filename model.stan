functions {
    real exponential_decay(
        real x,
        real alpha,
        real beta,
        real gamma
    ) {
        return 2 - exp(alpha*(pow(x, beta) - gamma));
    }
    real inv_sigmoid(
        real x,
        real alpha,
        real beta,
        real gamma
    ) {
        return gamma - (1 / (1 + exp(-alpha * (x - beta))));
    }
}
data {
    int<lower=0> T;             // total cycles observed
    int<lower=0> N;             // number of battery cells = 200
    int<lower=0> d;             // number of feature dimensions
    vector[T] y;
    vector[N] x1;
    vector[N] x2;
    vector[N] x3;
    array[N] int N_BC;          // cycle life for each battery cell
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

    //real gamma_0;
    //real gamma_1;
    //real gamma_2;
    //real gamma_3;

    real<lower=0> sigma;
}
transformed parameters {
    vector[N] alpha;
    vector[N] beta;
    vector[N] gamma;
    vector[T] y_hat;
{
    int idx = 1;
    real scaled_cycle_count;

    for(i in 1:N) {
        alpha[i] = alpha_0 + alpha_1*x1[i] + alpha_2*x2[i] + alpha_3*x3[i];
        beta[i] = beta_0 + beta_1*x1[i] + beta_2*x2[i] + beta_3*x3[i];
        gamma[i] = x1[i];

        for (j in 1:N_BC[i]) {
            scaled_cycle_count = j / 1000.0;
            //y_hat[idx] = exponential_decay(scaled_cycle_count, alpha[i], beta[i], gamma[i]);
            y_hat[idx] = inv_sigmoid(scaled_cycle_count, alpha[i], beta[i], gamma[i]);
            idx += 1;
        }
    }
}
}
model {
    alpha_0 ~ normal(2.5, 1);
    alpha_1 ~ normal(0, 1);
    alpha_2 ~ normal(0, 1);
    alpha_3 ~ normal(0, 1);

    beta_0 ~ normal(2.5, 1);
    beta_1 ~ normal(0, 1);
    beta_2 ~ normal(0, 1);
    beta_3 ~ normal(0, 1);

    //gamma_0 ~ normal(1.1, 1);
    //gamma_1 ~ normal(0, 1);
    //gamma_2 ~ normal(0, 1);
    //gamma_3 ~ normal(0, 1);

    sigma ~ gamma(1, 2);

    y ~ normal(y_hat, sigma);
}