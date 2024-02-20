library(tidyverse);  library(here); library(kableExtra); library(lme4)
library(brms); library(rptR); library(partR2); library(easystats)
library(ordinal); library(ggdist); library(tidyverse); library(ggthemes)
library(patchwork); library(cmdstanr); library(latex2exp)

set.seed(42) 

# 0. Simulate data ----

# Value storage
N_ID = 100
N_obs = 100
Group_size = 10
# Dataset structure
dfsim = data.frame(Obs = rep(1:N_obs, each = N_ID),
                   Id = rep(1:N_ID, N_obs), 
                   # Sample integers from 1 to 10 and repeat 100 times
                   Group = rep(sample.int(n = Group_size, size = Group_size,
                                          replace = F), N_obs),
                   Group_N = 10) %>% 
  mutate(Group_ID = paste(Group, Obs, sep = "_"))
dfsim %>% head(15) %>% kable()

set.seed(42) 

ID = data.frame(Id = 1:N_ID) %>% 
  mutate(pref = rnorm(n(), 0, 1))
dfsim = merge(dfsim, ID)

set.seed(42) 

dfsim = dfsim %>% 
  group_by(Group_ID) %>% 
  mutate(rank = rank(pref)) %>% 
  arrange(Obs, Group) %>% 
  ungroup()
dfsim %>% head(15) %>% kable()

# 1. Prior predictive checks ----
## 1.1 Define models and priors ----

bf.bin = bf(rank | trials(Group_N) ~ 1 + (1|Id),
            family = "binomial")

bf.poiss = bf(rank ~ 1 + (1|Id),
              family = "poisson")

bf.clmm = bf(rank ~ 1 + (1|Id),
             family = "cumulative")

priors.bin <- 
  # Intercepts priors
  prior(normal(.2, .05), class = Intercept) +
  # Random effects priors (default to exp(1))
  prior(exponential(1), class = sd)

priors.poiss <- 
  # Intercepts priors
  prior(normal(1.6, .1), class = Intercept, lb = 0) +
  # Random effects priors
  prior(exponential(1), class = sd)

priors.clmm <- 
  # Intercepts priors
  prior(normal(0, 1), class = Intercept) +
  # Random effects priors (default to exp(1))
  prior(exponential(1), class = sd)


priors.bin %>% 
  parse_dist() %>% 
  # filter(class == "b") %>% 
  ggplot(aes(xdist = .dist_obj, y = format(.dist_obj))) +
  stat_dist_halfeye() +
  facet_wrap(~class, scales = "free") +
  ggtitle("Prior distribution") +
  xlab("Value") + ylab("Density") +
  theme_bw(12) +
  theme(axis.text.y = element_text(angle = 90)) 

priors.poiss %>% 
  parse_dist() %>% 
  # filter(class == "b") %>% 
  ggplot(aes(xdist = .dist_obj, y = format(.dist_obj))) +
  stat_dist_halfeye() +
  facet_wrap(~class, scales = "free") +
  ggtitle("Prior distribution") +
  xlab("Value") + ylab("Density") +
  theme_bw(12) +
  theme(axis.text.y = element_text(angle = 90)) 

priors.clmm %>% 
  parse_dist() %>% 
  # filter(class == "b") %>% 
  ggplot(aes(xdist = .dist_obj, y = format(.dist_obj))) +
  stat_dist_halfeye() +
  facet_wrap(~class, scales = "free") +
  ggtitle("Prior distribution") +
  xlab("Value") + ylab("Density") +
  theme_bw(12) +
  theme(axis.text.y = element_text(angle = 90)) 

## 1.2 Fit and store models ----
brms.bin.prior = brm(bf.bin,
                     data = dfsim,
                     prior = priors.bin,
                     sample_prior = "only",
                     warmup = 1000,
                     iter = 2000,
                     seed = 42, 
                     cores = 8, 
                     threads = threading(8),
                     control = list(adapt_delta = .99,
                                    max_treedepth = 15),
                     backend = "cmdstanr", 
                     file_refit = "always",
                     file = here("outputs/mods/brms.bin.prior"))

brms.poiss.prior = brm(bf.poiss, 
                       data = dfsim,
                       prior = priors.poiss,
                       sample_prior = "only",
                       warmup = 1000,
                       iter = 2000,
                       seed = 42, 
                       cores = 8, 
                       threads = threading(8),
                       control = list(adapt_delta = .99,
                                      max_treedepth = 15),
                       backend = "cmdstanr", 
                       file_refit = "always",
                       file = here("outputs/mods/brms.poiss.prior"))

brms.clmm.prior = brm(bf.clmm, 
                      data = dfsim,
                      prior = priors.clmm,
                      sample_prior = "only",
                      warmup = 1000,
                      iter = 2000,
                      seed = 42, 
                      cores = 8,
                      threads = threading(8),
                      control = list(adapt_delta = .99,
                                     max_treedepth = 15),
                      backend = "cmdstanr", 
                      file_refit = "always",
                      file = here("outputs/mods/brms.clmm.prior"))

brms.bin.prior; brms.poiss.prior; brms.clmm.prior

## 1.3 Plot ----

plot.bin = pp_check(brms.bin.prior, ndraws = 500) +
  xlim(0, 15) +
  ggtitle("Binomial GLMM") 

plot.poiss = pp_check(brms.poiss.prior, ndraws = 500) +
  xlim(0, 15) +
  ggtitle("Poisson GLMM") 

plot.clmm = pp_check(brms.clmm.prior, ndraws = 500) +
  xlim(0, 15) +
  ggtitle("Cumulative Link Mixed Model")


plot.priorpcheck = (plot.bin / plot.poiss / plot.clmm) +
  plot_annotation("Prior-predictive checks") &
  theme_bw(14)

plot.priorpcheck

ggsave(filename = "outputs/figs/plot.priorpcheck.jpeg", plot.priorpcheck)


# 2. Fit model to simulated data ----
brms.bin = brm(bf.bin,
                     data = dfsim,
                     prior = priors.bin,
                     warmup = 5000,
                     iter = 6000,
                     seed = 42, 
                     cores = 8, 
                     threads = threading(8),
                     control = list(adapt_delta = .99,
                                    max_treedepth = 15),
                     backend = "cmdstanr", 
                     file_refit = "always",
                     file = here("outputs/mods/brms.bin"))

brms.poiss = brm(bf.poiss, 
                       data = dfsim,
                       prior = priors.poiss,
                       warmup = 5000,
                       iter = 6000,
                       seed = 42, 
                       cores = 8, 
                       threads = threading(8),
                       control = list(adapt_delta = .99,
                                      max_treedepth = 15),
                       backend = "cmdstanr", 
                       file_refit = "always",
                       file = here("outputs/mods/brms.poiss"))

brms.clmm = brm(bf.clmm, 
                      data = dfsim,
                      prior = priors.clmm,
                      warmup = 5000,
                      iter = 6000,
                      seed = 42, 
                      cores = 8,
                      threads = threading(8),
                      control = list(adapt_delta = .99,
                                     max_treedepth = 15),
                      backend = "cmdstanr", 
                      file_refit = "always",
                      file = here("outputs/mods/brms.clmm"))

brms.bin
brms.poiss
brms.clmm

plot(brms.bin); plot(brms.poiss); plot(brms.clmm) 

waic(brms.clmm, brms.bin, brms.poiss)

## 2.1 Posterior predictive checks ----
plot.bin = pp_check(brms.bin, ndraws = 500) +
  xlim(0, 15) +
  ggtitle("Binomial GLMM") 

plot.poiss = pp_check(brms.poiss, ndraws = 500) +
  xlim(0, 15) +
  ggtitle("Poisson GLMM") 

plot.clmm = pp_check(brms.clmm, ndraws = 500) +
  xlim(0, 15) +
  ggtitle("Cumulative Link Mixed Model")


plot.ppcheck = (plot.bin / plot.poiss / plot.clmm) +
  plot_annotation("Posterior-predictive checks") &
  theme_bw(14)

plot.ppcheck

ggsave(filename = "outputs/figs/plot.ppcheck.jpeg", plot.ppcheck)


## 2.2 Variance components comparison ----
Vi.clmm = brms.clmm %>% 
  spread_draws(sd_Id__Intercept)

Vi.bin = brms.bin %>% 
  spread_draws(sd_Id__Intercept)

Vi.poiss = brms.poiss %>% 
  spread_draws(sd_Id__Intercept)

df.Vi = data.frame(
  Vi = c(Vi.clmm$sd_Id__Intercept, 
         Vi.bin$sd_Id__Intercept,
         Vi.poiss$sd_Id__Intercept),
  Mod = factor(c(rep("CLMM", length(Vi.clmm$sd_Id__Intercept)),
                 rep("Binomial", length(Vi.bin$sd_Id__Intercept)),
                 rep("Poisson", length(Vi.poiss$sd_Id__Intercept))),
               levels = c("CLMM", "Binomial", "Poisson")))

plot.Vi = df.Vi %>% 
  ggplot(aes(x = Vi, y = Mod, fill = Mod)) +
  stat_halfeye() + 
  scale_fill_wsj() +
  xlab("Among-individual variance") +
  ylab("Model") +
  theme_bw(14) +
  theme(legend.position = "none")
plot.Vi

ggsave(filename = "outputs/figs/plot.Vi.jpeg", plot.Vi)


## 2.3 BLUPS comparison
# install.packages("devtools")
devtools::install_github("m-clark/visibly")
p1 = plot_coefficients(brms.bin, ranef = TRUE, which_ranef = 'Id')$`(Intercept)`
p2 = plot_coefficients(brms.poiss, ranef = TRUE, which_ranef = 'Id')$`(Intercept)`
p3 = plot_coefficients(brms.clmm, ranef = TRUE, which_ranef = 'Id')$`(Intercept)`


# 3. Leading/Following GLMM ----
dfsim = dfsim %>% 
  mutate(lead = case_when(rank == 1 ~ 1, 
                        rank > 1 ~ 0))
ID$Sex <- as.factor(c(rep("F", nrow(ID)/2),
                      rep("M", nrow(ID)/2)))
dfsim = merge(dfsim, ID)

## 3.1 brms ----

bf.bern = bf(lead ~ 1 + (1|Id),
            family = "bernoulli")
priors.bin <- 
  # Intercepts priors
  prior(normal(0, 1.5), class = Intercept) +
  # Random effects priors (default to exp(1))
  prior(exponential(1), class = sd)

brms.bern.prior = brm(bf.bern,
                     data = dfsim,
                     prior = priors.bin,
                     sample_prior = "only",
                     warmup = 1000,
                     iter = 2000,
                     seed = 42, 
                     cores = 8, 
                     threads = threading(8),
                     control = list(adapt_delta = .99,
                                    max_treedepth = 15),
                     backend = "cmdstanr", 
                     file_refit = "always",
                     file = here("outputs/mods/brms.bern.prior"))

pp_check(brms.bern.prior, ndraws = 500)

brms.bern = brm(bf.bern,
                data = dfsim,
                prior = priors.bin,
                warmup = 1000,
                iter = 2000,
                seed = 42, 
                cores = 8, 
                threads = threading(8),
                control = list(adapt_delta = .99,
                               max_treedepth = 15),
                backend = "cmdstanr", 
                file_refit = "always",
                file = here("outputs/mods/brms.bern"))

pp_check(brms.bern, ndraws = 500)
r2(brms.bern)

bf.bern.2 = bf(lead ~ 0 + Sex + (0 + Sex||Id),
               family = "bernoulli")
priors.bern <- 
  # Intercepts priors
  prior(normal(0, 1.5), class = b) +
  # Random effects priors (default to exp(1))
  prior(exponential(1), class = sd)

brms.bern.2 = brm(bf.bern.2,
                data = dfsim,
                prior = priors.bern,
                warmup = 1000,
                iter = 2000,
                seed = 42, 
                cores = 8, 
                threads = threading(8),
                control = list(adapt_delta = .99,
                               max_treedepth = 15),
                backend = "cmdstanr", 
                file_refit = "always",
                file = here("outputs/mods/brms.bern.2"))

pp_check(brms.bern.2, ndraws = 500)


## 3.2 rptR ----
# Adjusted repeatability
rpt.R.f = rpt(formula = lead ~ 1 + (1|Id), 
            grname = "Id", 
            datatype = "Binary", 
            data = subset(dfsim, Sex == "F"))
rpt.R.m = rpt(formula = lead ~ 1 + (1|Id), 
            grname = "Id", 
            datatype = "Binary", 
            data = subset(dfsim, Sex == "M"))
saveRDS(rpt.R.f, here("outputs/mods/rpt.R.f.bin.rds"))
saveRDS(rpt.R.m, here("outputs/mods/rpt.R.m.bin.rds"))

# All variance components
rpt.V.f <- rpt(formula = lead ~ 1 + (1|Id), 
               grname = c("Id", "Fixed", "Residual"), 
               datatype = c("Binary"), 
               data = subset(dfsim, Sex == "F"),
               ratio = FALSE)
rpt.V.m <- rpt(formula = lead ~ 1 + (1|Id), 
               grname = c("Id", "Fixed", "Residual"), 
               datatype = "Binary", 
               data = subset(dfsim, Sex == "M"),
               ratio = FALSE)

saveRDS(rpt.V.f, here("outputs/mods/rpt.V.f.bin.rds"))
saveRDS(rpt.V.m, here("outputs/mods/rpt.V.m.bin.rds"))

# Store all vectors of bootstrapped values
# Load models
rpt.R.f = read_rds(here("outputs/mods/rpt.R.f.bin.rds"))
rpt.R.m = read_rds(here("outputs/mods/rpt.R.m.bin.rds"))
rpt.V.f = read_rds(here("outputs/mods/rpt.V.f.bin.rds"))
rpt.V.m = read_rds(here("outputs/mods/rpt.V.m.bin.rds"))

plot(rpt.R.f)
plot(rpt.R.m)
plot(rpt.V.f)
plot(rpt.V.m)


Vi_f <- rpt.V.f$R_boot_link$Id
Vi_m <- rpt.V.m$R_boot_link$Id
Vfe_f <- rpt.V.f$R_boot_link$Fixed
Vfe_m <- rpt.V.m$R_boot_link$Fixed
VR_f <- rpt.V.f$R_boot_link$Residual
VR_m <- rpt.V.m$R_boot_link$Residual
R_f <- rpt.R.f$R_boot_link$Id
R_m <- rpt.R.m$R_boot_link$Id

df <- data.frame(Vi = c(Vi_f, Vi_m),
                 Vfe = c(Vfe_f, Vfe_m),
                 VR = c(VR_f, VR_m),
                 R = c(R_f, R_m),
                 Sex = c(rep("F", length(Vi_f)),
                         rep("M", length(Vi_m)))) # %>% 
  # Long format
  # pivot_longer(cols = Vi:R, 
  #              names_to = "v.compo", 
  #              values_to = "var")


# Store effect sizes
df.2  <- data.frame(delta_Vi = Vi_f - Vi_m,
                    delta_Vfe = Vfe_f - Vfe_m,
                    delta_VR = VR_f - VR_m,
                    delta_R = R_f - R_m) #%>% 
  # Long format
  # pivot_longer(cols = delta_Vi:delta_R, 
  #              names_to = "d.v.compo", 
  #              values_to = "d.var")


p1 = df %>% 
  ggplot(aes(x = Vi, fill = Sex)) +
  stat_halfeye(alpha = .6) + 
  scale_fill_wsj() +
  xlab(bquote("Among-individual variance ("*V[i]*")")) +
  ylab("Density") +
  theme_bw(14)
delta.p1 = df.2 %>% 
  ggplot(aes(x = delta_Vi)) +
  stat_halfeye(alpha = .6) + 
  xlab(bquote(Delta[V[i]])) +
  ylab("Density") +
  theme_bw(14)
p1 = p1 + delta.p1

p2 = df %>% 
  ggplot(aes(x = Vfe, fill = Sex)) +
  stat_halfeye(alpha = .6) + 
  scale_fill_wsj() +
  xlab(bquote("Fixed effect variance ("*V[fe]*")")) +
  ylab("Density") +
  theme_bw(14)
delta.p2 = df.2 %>% 
  ggplot(aes(x = delta_Vfe)) +
  stat_halfeye(alpha = .6) + 
  xlab(bquote(Delta[V[fe]])) +
  ylab("Density") +
  theme_bw(14)
p2 = p2 + delta.p2


p3 = df %>% 
  ggplot(aes(x = Vfe, fill = Sex)) +
  stat_halfeye(alpha = .6) + 
  scale_fill_wsj() +
  xlab(bquote("Residual variance ("*V[R]*")")) +
  ylab("Density") +
  theme_bw(14)
delta.p3 = df.2 %>% 
  ggplot(aes(x = delta_VR)) +
  stat_halfeye(alpha = .6) + 
  xlab(bquote(Delta[V[fe]])) +
  ylab("Density") +
  theme_bw(14)
p3 = p3 + delta.p3

p4 = df %>% 
  ggplot(aes(x = R, fill = Sex)) +
  stat_halfeye(alpha = .6) + 
  scale_fill_wsj() +
  xlim(0, 1) +
  xlab(bquote("Repeatability (R)")) +
  ylab("Density") +
  theme_bw(14)
delta.p4 = df.2 %>% 
  ggplot(aes(x = delta_R)) +
  stat_halfeye(alpha = .6) + 
  xlim(0, 1) +
  xlab(bquote(Delta[R])) +
  ylab("Density") +
  theme_bw(14)
p4 = p4 + delta.p4

plot_var_R = p1 / p2 / p3 / p4
plot_var_R

ggsave(filename = "outputs/figs/plot_var_R.jpeg", plot_var_R)

