#install.packages(c('forecast', 'tidyverse', 'zoo', 'devtools',
#                   'ggplot2', 'seasonal', 'MASS', 'urca', 'MTS',
#                   'vars', 'lmtest', 'tseries', 'readxl', 
#                   'mFilter', 'gets', 'FinTS', 'ggpubr', l'pirfs', 'glue',
#                    'gridExtra'))

library('gridExtra')
library('ggplot2')
library('tidyverse')
library('zoo')
library('devtools')
library('ggplot2')
library('seasonal')
library('MASS')
library('urca')
library('MTS')
library('vars')
library('lmtest')
library('tseries')
library('readxl')
library('mFilter')
library('gets')
library('FinTS')
library('ggpubr')
library('forecast')
library('lpirfs')
library('ggplot2')
library('glue')

setwd("D:\\OneDrive\\0 - ATESE\\Dados")

### Importar dados
PATH_OUTPUT_DADOS_ECONOMICOS <- "D:\\OneDrive\\0 - ATESE\\Dados\\output_dados_economicos\\df_var.csv"
df <- read.csv(PATH_OUTPUT_DADOS_ECONOMICOS)


# ---- Script ---- 
### Definir série temporal
dados_ts <- ts(df, start = c(2009, 1), frequency = 12)

# ##Definir variáveis

#### Modelo Proposto
vix <- dados_ts[, 'ln_dessaz_indice_vix_mean_close']
sent <- dados_ts[, 'dessaz_normalized_sentiment_weighted_agg_carosia_by_month_with_op']
##### Bloco variáveis reais
fbcf <- dados_ts[, 'ln_dessaz_indice_100_indic_ipea_fbcf']
ibcbr <- dados_ts[, 'ln_dessaz_indice_100_ibc_br']
emprego <- dados_ts[, 'ln_dessaz_estoque_emprego']
ipca <- dados_ts[, 'ln_dessaz_ipca_index']
##### Bloco variáveis financeiras
tx_prefix_juros <- dados_ts[, 'ln_dessaz_tx_prefix_media_juros']
ibov <- dados_ts[, 'ln_dessaz_ibov_index']
ptax <- dados_ts[, 'ln_dessaz_ptax']
tx_selic_meta <- dados_ts[, 'ln_dessaz_tx_selic_meta']
##### outras var
tx_ipca = dados_ts[, 'dessaz_tx_ipca'] # não faz sentido tirar log pq tem negativo


# O Modelo

## Análise Gráfica
par(mfrow(c(3,1)))
plot(vix)
plot(sent)
plot(fbcf)
plot(ipca)



## FAC e FACP
acf(vix, lag.max=36, main = 'FAC', xlab='defasagem', ylab='autocorrelações')
acf(sent, lag.max=36, main = 'FAC', xlab='defasagem', ylab='autocorrelações')
acf(fbcf, lag.max=36, main = 'FAC', xlab='defasagem', ylab='autocorrelações')

pacf(vix, lag.max=36, main = 'FAC', xlab='defasagem', ylab='autocorrelações')
pacf(sent, lag.max=36, main = 'FAC', xlab='defasagem', ylab='autocorrelações')
pacf(fbcf, lag.max=36, main = 'FAC', xlab='defasagem', ylab='autocorrelações')

## Decomposição sazonal
plot(decompose(vix))


# Testes de Raiz Unitária 
# Aug-dikey-fuller: estatística à direita do valor crítico -> tem raiz unitária
# H0: tem raiz unitária

## Vix -
summary(ur.df(vix, type = c("trend"), lags = 8, selectlags =  'BIC')) # estacionaria
summary(ur.df(sent, type = c("drift"), lags = 8, selectlags =  'BIC')) # estacionaria
summary(ur.df(fbcf, type = c("none"), lags = 8, selectlags =  'BIC')) # n-estacionaria
summary(ur.df(ibcbr, type = c("none"), lags = 8, selectlags =  'BIC'))# n-estacionaria
summary(ur.df(emprego, type = c("none"), lags = 8, selectlags =  'BIC')) # n-estacionaria
summary(ur.df(tx_prefix_juros, type = c("none"), lags = 8, selectlags =  'BIC')) # n-estacionaria
summary(ur.df(ipca, type = c("none"), lags = 8, selectlags =  'BIC')) # n-estacionaria
summary(ur.df(ibov, type = c("none"), lags = 8, selectlags =  'BIC')) # n-estacionaria
summary(ur.df(ptax, type = c("none"), lags = 8, selectlags =  'BIC')) # n-estacionaria

# apenas taxa selic meta não é estacionário à primeira diferença (mas ln dá)
summary(ur.df(diff(tx_selic_meta), type = c("trend"), lags = 8, selectlags =  'BIC'))
summary(ur.df(diff(tx_selic_meta), type = c("drift"), lags = 8, selectlags =  'BIC'))
summary(ur.df(diff(tx_selic_meta), type = c("none"), lags = 8, selectlags =  'BIC'))



# Seleção da ordem do VAR 
####dados_df <- data.frame(cbind(vix, sent, fbcf, ibcbr, emprego, ipca, tx_prefix_juros, ibov, ptax, tx_selic_meta))
#### Modelo Ajustado com Considerações Rafael (usando 1ª diferença)
dados_df <- data.frame(cbind(#vix, 
                             diff(sent), 
                             diff(fbcf), diff(ibcbr), diff(tx_ipca), 
                             diff(tx_prefix_juros), diff(ibov), diff(ptax)))


dados_var <- ts(dados_df, start = c(2009, 2), frequency = 12)
 
p <- VARselect(dados_var, exogen = diff(vix), #season = 12 # já passei filtro de sazonalidade
              lag.max = 6, type= 'const'); p #### p = 2 pelo AIC


# Estimação do modelo 
vix_df <- data.frame(vix = diff(vix)) # entra como exógeno
var <- vars::VAR(dados_var, exogen = vix_df, # season = 12,
                 p = 1, type = 'const')
summary(var)


# Verificação do modelo 

## Análise de estabilidade
roots(var) # verificar se todas são menor do que 1 em módulo


## FAC resíduos do modelo

acf(residuals(var)[,1], main='Resíduo - Equação 1') # verificar ordem no modelo
acf(residuals(var)[,2], main='Resíduo - Equação 2')
acf(residuals(var)[,3], main='Resíduo - Equação 3')


## Teste Arch-lm Multivariado     H0: os resíduos nao possuem efeitos auto-regressivos de heterocedasticidade condicional
var.arch <- arch.test(var, lags.multi = 8); var.arch # p alto, não rejeto hipotese nula - ta ok se for alto

## Teste de Correlação Serial     H0: os resíduos são iid
var.pt.asy <- serial.test(var, lags.pt = 8, type = 'PT.asymptotic'); var.pt.asy # Teste de Portmanteau - grandes amostras
var.pt.adj <- serial.test(var, lags.pt = 8, type = 'PT.adjusted'); var.pt.adj # Teste de Portmanteau - pequenas amostras
var.BG=serial.test(var, lags.pt=8, type='BG'); var.BG                           # Teste LM - Proposto por Breusch-Godfrey
var.ES=serial.test(var, lags.pt=8, type='ES'); var.ES                           # Teste F - Proposto por Edgerton-Shukur

## Teste de Normalidade --- Teste de Jarque-Bera | H0: normalidade dos resíduos

### Univariado 
var$varresult$diff.sent.

e_sent <- var$varresult$diff.sent$residuals # residuos da equação VIX
par(mfrow=c(2,2))
hist(e_sent, freq=F, ylab='Densidade', xlab='Resíduos', main='Residuos')
plot(density(e_sent, kernel = c('gaussian')), main='Resíduos') # função densidade estimada
qqnorm(e_sent, ylab = 'Quantis amostrais', xlab='Quantis teóricos', main='Quantil-Quantil')
qqline(e_sent, col = 'red')


shapiro.test(e_sent) # p baixo REJEITO H0: normalidade dos resíduos
jarque.bera.test(e_sent)

### Multivariado
var.norm <- normality.test(var, multivariate.only = FALSE); var.norm
plot(var.norm)



# Aplicações do MODELO 

# Função de Resposta ao Impulso (FRI) 

## FRI padrão
var.irf <- irf(var, n.ahead = 12, 
               ortho = T, cumulative = F,
               boot = TRUE, runs = 500); var.irf

par(mfcol=c(1,3), cex = 0.6)
plot(var.irf, plot.type='single')


## Alterando magnitude do choque e especificando impulso e resposta
irf_sent <- irf(var, impulse = 'diff.sent.',
                     #response = 'diff.tx_prefix_juros.',
                     n.ahead = 12, ortho = T,
                     cumulative = F, boot = TRUE, runs = 500)

plot(irf_sent) # ERROR figure margins too large --> ajustar plot_area à direita



irf_sent # irf, lower, upper
irf_sent$irf # só irf
irf_sent$irf$diff.sent # como é apenas 1 impulso, tem apenas 1 lista de valores 
irf_sent$irf$diff.sent[, "diff.sent."] # onde dá para selecionar diferentes colunas


names(var.irf$irf) # nomes das colunas - impulsos possíveis

### Apenas 1 impulso e 1 resposta 
irf_sent_juros <- irf(var, impulse = 'diff.sent.',
                response = 'diff.ptax.',
                n.ahead = 12, ortho = T,
                cumulative = F, boot = TRUE, runs = 500); plot(irf_sent_juros)

names(var.irf$irf)






# O Modelo que Rodei no python (mas vix sendo exóg) -----------------
vix <- dados_ts[, 'ln_dessaz_indice_vix_mean_close']
sent <- dados_ts[, 'dessaz_normalized_sentiment_weighted_agg_carosia_by_month_with_op']
fbcf <- dados_ts[, 'ln_dessaz_indice_100_indic_ipea_fbcf']
ibcbr <- dados_ts[, 'ln_dessaz_indice_100_ibc_br']
emprego <- dados_ts[, 'ln_dessaz_estoque_emprego']
ipca <- dados_ts[, 'ln_dessaz_ipca_index']
tx_prefix_juros <- dados_ts[, 'ln_dessaz_tx_prefix_media_juros']
ibov <- dados_ts[, 'ln_dessaz_ibov_index']
ptax <- dados_ts[, 'ln_dessaz_ptax']
tx_selic_meta <- dados_ts[, 'ln_dessaz_tx_selic_meta']
tx_ipca = dados_ts[, 'dessaz_tx_ipca']

dados_df <- data.frame(cbind(sent,
                             fbcf, ibcbr, emprego, ipca,
                             tx_prefix_juros, ibov, ptax, tx_selic_meta))

vix_df <- data.frame(vix = vix)

dados_var <- ts(dados_df, start = c(2009, 2), frequency = 12)

p <- VARselect(dados_var, exogen = vix, #season = 12 # já passei
               lag.max = 6, type= 'const'); p

var <- vars::VAR(dados_var, exogen = vix_df, 
                 p = 4, type = 'const')
roots(var)

names(dados_df)

### Funções impulso resposta individuais
impulse_var <- 'sent'
N_AHEAD <- 30

irf_sent <- irf(var, impulse = impulse_var, #'sent.',
                response = 'fbcf',
                n.ahead = N_AHEAD, ortho = T,
                cumulative = F, boot = TRUE, runs = 500); plot(irf_sent)

irf_sent <- irf(var, impulse = impulse_var,
                response = 'ibcbr',
                n.ahead = N_AHEAD, ortho = T,
                cumulative = F, boot = TRUE, runs = 500); plot(irf_sent)

irf_sent <- irf(var, impulse = impulse_var,
                response = 'emprego',
                n.ahead = N_AHEAD, ortho = T,
                cumulative = F, boot = TRUE, runs = 500); plot(irf_sent)

irf_sent <- irf(var, impulse = impulse_var,
                response = 'ipca',
                n.ahead = N_AHEAD, ortho = T,
                cumulative = F, boot = TRUE, runs = 500); plot(irf_sent)

irf_sent <- irf(var, impulse = impulse_var,
                response = 'tx_prefix_juros',
                n.ahead = N_AHEAD, ortho = T,
                cumulative = F, boot = TRUE, runs = 500); plot(irf_sent)

irf_sent <- irf(var, impulse = impulse_var,
                response = 'ibov',
                n.ahead = N_AHEAD, ortho = T,
                cumulative = F, boot = TRUE, runs = 500); plot(irf_sent)

irf_sent <- irf(var, impulse = impulse_var,
                response = 'ptax',
                n.ahead = N_AHEAD, ortho = T,
                cumulative = F, boot = TRUE, runs = 500); plot(irf_sent)

irf_sent <- irf(var, impulse = impulse_var,
                response = 'tx_selic_meta',
                n.ahead = N_AHEAD, ortho = T,
                cumulative = F, boot = TRUE, runs = 500); plot(irf_sent)

#### Objeto IRF
IMPULSE_VAR <- 'sent'
N_AHEAD <- 30
var_irf_sent <- irf(var, n.ahead = N_AHEAD, 
                    #impulse_var = IMPULSE_VAR,
                    ortho = T, cumulative = F,
                    boot = TRUE, runs = 500); var_irf_sent

var_irf_sent # irf, lower, upper
var_irf_sent$irf # só irf
var_irf_sent$irf$sent # irf do sent nas demais 
var_irf_sent$irf$sent[, "fbcf"] # irf do sent na fbcf


var_fbcf_mean <- var_irf_sent$irf$sent[, "fbcf"] 
var_fbcf_lower <- var_irf_sent$Lower$sent[, "fbcf"] 
var_fbcf_upper <- var_irf_sent$Upper$sent[, "fbcf"] 

df <- data.frame(
  index = 0:N_AHEAD,
  var_fbcf_mean = var_fbcf_mean,
  var_fbcf_lower = var_fbcf_lower,
  var_fbcf_upper = var_fbcf_upper)
df

plot <- ggplot(df, aes(x = 0:N_AHEAD)) +
  geom_ribbon(aes(ymin = var_fbcf_lower, ymax = var_fbcf_upper), fill = "gray", alpha = 0.5) +
  geom_line(aes(y = var_fbcf_mean), color = "black") +
  geom_line(aes(y = var_fbcf_lower), linetype = "dashed", color = "gray") +
  geom_line(aes(y = var_fbcf_upper), linetype = "dashed", color = "gray") +
  labs(title = "Line Plot of Response Mean",
       x = "X-axis Label",
       y = "Response Mean")

plot



# Local Projections ------------------------------------------------------------
dados_ts <- ts(df, start = c(2009, 1), frequency = 12)

vix <- dados_ts[, 'ln_dessaz_indice_vix_mean_close']
sent <- dados_ts[, 'dessaz_normalized_sentiment_weighted_agg_carosia_by_month_with_op']
fbcf <- dados_ts[, 'ln_dessaz_indice_100_indic_ipea_fbcf']
ibcbr <- dados_ts[, 'ln_dessaz_indice_100_ibc_br']
emprego <- dados_ts[, 'ln_dessaz_estoque_emprego']
ipca <- dados_ts[, 'ln_dessaz_ipca_index']
tx_prefix_juros <- dados_ts[, 'ln_dessaz_tx_prefix_media_juros']
ibov <- dados_ts[, 'ln_dessaz_ibov_index']
ptax <- dados_ts[, 'ln_dessaz_ptax']
tx_selic_meta <- dados_ts[, 'ln_dessaz_tx_selic_meta']
tx_ipca = dados_ts[, 'dessaz_tx_ipca']

dados_df <- data.frame(cbind(#vix,
                            sent,
                            fbcf, ibcbr, emprego, 
                            ipca,
                            #tx_ipca,
                            tx_prefix_juros, ibov, ptax, tx_selic_meta))
dados_vix <- data.frame(vix = as.vector(vix))

# define the parameters
lags <- 4
horizon <- 30

results_lin <- lp_lin(
              endog_data = dados_df,
              lags_endog_lin = lags,
              trend = 0,
              shock_type = 1,  # 1 for a unit shock to the first variable, 2 for the second, etc.
              confint = 1.96,
              exog_data = dados_vix,
              lags_exog = 4, 
              contemp_data = NULL,  # no contemporaneous variables in this example
              hor = horizon
              )

plot(results_lin)

# OLS diagnostics for the first shock of the first horizon
summary(results_lin)[[1]][1]

# irf object structure
results_lin$irf_lin_mean # a 3d array
results_lin$irf_lin_mean[, , 1]
# returns a KXH matrix, K = nº vars, H = nº hor, ’1’ is the shock variable, 
# corresponding to the first variable in endog_data.
results_lin$irf_lin_mean[1, , 1]



# Var x Lp ------------------------------------------------------------ 
## var - modelo ln-nível (o q rodei no python, mas com vix exog)
dados_df <- data.frame(cbind(sent,
                             fbcf, ibcbr, emprego, ipca,
                             tx_prefix_juros, ibov, ptax, tx_selic_meta))
vix_df <- data.frame(vix = vix)
dados_var <- ts(dados_df, start = c(2009, 1), frequency = 12)
p <- VARselect(dados_var, exogen = vix, #season = 12 # já passei
               lag.max = 6, type= 'const'); p
var <- vars::VAR(dados_var, exogen = vix_df, 
                 p = 4, type = 'const')
roots(var)


IMPULSE_VAR <- 'sent'
N_AHEAD <- 30
var_irf_sent <- irf(var, n.ahead = N_AHEAD, 
                    #impulse_var = IMPULSE_VAR,
                    ortho = T, cumulative = F,
                    boot = TRUE, runs = 500); var_irf_sent

var_fbcf_mean <- var_irf_sent$irf$sent[,2] # 2 = fbcf
var_fbcf_mean <- var_irf_sent$irf$sent[, "fbcf"] 
var_fbcf_lower <- var_irf_sent$Lower$sent[, "fbcf"] 
var_fbcf_upper <- var_irf_sent$Upper$sent[, "fbcf"] 
var_fbcf_mean

df <- data.frame(
  index = 0:N_AHEAD,
  var_fbcf_mean = var_fbcf_mean,
  var_fbcf_lower = var_fbcf_lower,
  var_fbcf_upper = var_fbcf_upper)
df

plot <- ggplot(df, aes(x = 0:N_AHEAD)) +
  geom_ribbon(aes(ymin = var_fbcf_lower, ymax = var_fbcf_upper), fill = "gray", alpha = 0.5) +
  geom_line(aes(y = var_fbcf_mean), color = "black") +
  geom_line(aes(y = var_fbcf_lower), linetype = "dashed", color = "gray") +
  geom_line(aes(y = var_fbcf_upper), linetype = "dashed", color = "gray") +
  labs(title = "Line Plot of Response Mean",
       x = "X-axis Label",
       y = "Response Mean")

plot


## lp
dados_df <- data.frame(cbind(sent,
                             fbcf, ibcbr, emprego, ipca,
                             tx_prefix_juros, ibov, ptax, tx_selic_meta))
dados_vix <- data.frame(vix = as.vector(vix))

lags <- 4
horizon <- 30

results_lin <- lp_lin(
    endog_data = dados_df,
    lags_endog_lin = lags,
    trend = 0,
    shock_type = 1,  # 1 for a unit shock to the first variable, 2 for the second, etc.
    confint = 1.96,
    exog_data = dados_vix,
    lags_exog = 4, 
    contemp_data = NULL,  # No contemporaneous variables in this example
    hor = horizon
)

plot(results_lin)

lp_fbcf_mean <- results_lin$irf_lin_mean[2, 1:horizon , 1]
lp_fbcf_lower <- results_lin$irf_lin_low[2, 1:horizon , 1]
lp_fbcf_up <- results_lin$irf_lin_up[2, 1:horizon , 1]
lp_fbcf_mean

df <- data.frame(
  index = 1:length(lp_fbcf_mean),
  lp_fbcf_mean = lp_fbcf_mean,
  lp_fbcf_lower = lp_fbcf_lower,
  lp_fbcf_up = lp_fbcf_up)

plot <- ggplot(df, aes(x = 1:horizon)) +
  geom_ribbon(aes(ymin = lp_fbcf_lower, ymax = lp_fbcf_up), fill = "gray", alpha = 0.5) +
  geom_line(aes(y = lp_fbcf_mean), color = "black") +
  geom_line(aes(y = lp_fbcf_lower), linetype = "dashed", color = "gray") +
  geom_line(aes(y = lp_fbcf_up), linetype = "dashed", color = "gray") +
    labs(title = "Line Plot of Response Mean",
         x = "X-axis Label",
         y = "Response Mean")

plot


### plot var x lp 

#### store column labels and positions
response_var_names <- names(var_irf_sent$irf)
response_var_indexes <- 1:9
response_var_names_indexes <- mapply(
    function(x, y) list(x, y), response_var_names,
                               response_var_indexes, 
    SIMPLIFY = FALSE)

results_df <- data.frame(
  name = character(),
  index = integer(),
  t = integer(),
  lp_mean = numeric(),
  lp_lower = numeric(),
  lp_upper = numeric(),
  var_mean = numeric(),
  var_lower = numeric(),
  var_upper = numeric(),
  stringsAsFactors = FALSE)


#### loop
for (pair in response_var_names_indexes) {
  for (t in 1:horizon) {
    name <- pair[[1]]
    index <- pair[[2]]
    # lp
    lp_mean <- results_lin$irf_lin_mean[index, t , 1]
    lp_lower <- results_lin$irf_lin_low[index, t , 1]
    lp_up <- results_lin$irf_lin_up[index, t , 1]
    # var
    var_mean <- var_irf_sent$irf$sent[t, index] 
    var_lower <- var_irf_sent$Lower$sent[t, index] 
    var_upper <- var_irf_sent$Upper$sent[t, index] 
    # populate df
    temp_df <- data.frame(
    name = name,
    index = index,
    t = t,
    lp_mean = lp_mean,
    lp_lower = lp_lower,
    lp_upper = lp_up,
    var_mean = var_mean,
    var_lower = var_lower,
    var_upper = var_upper,
    stringsAsFactors = FALSE)
    # append the temporary data frame to the main data frame
    results_df <- rbind(results_df, temp_df)
  }
}

results_df

plot_list <- list()

for (VAR_NAME in unique(results_df$name)) {
  plot <- ggplot(subset(results_df, name == VAR_NAME), 
                 aes(x = t)) +
    # lp
    geom_ribbon(aes(ymin = lp_lower, ymax = lp_upper), fill = "gray", alpha = 0.5) +
    geom_line(aes(y = lp_mean), color = "black", linewidth = 0.5) +
    geom_line(aes(y = lp_lower), color = "gray", linewidth = 0.5) +
    geom_line(aes(y = lp_upper),  color = "gray", linewidth = 0.5) +
    # var
    geom_ribbon(aes(ymin = var_lower, ymax = var_upper), fill = "#FF4500", alpha = 0.3) +
    geom_line(aes(y = var_mean), linetype = "longdash", color = "#FF4500", linewidth = 0.5) +
    geom_line(aes(y = var_lower), linetype = "longdash", color = "#FF4500", alpha = 0.5, linewidth = 0.5) +
    geom_line(aes(y = var_upper), linetype = "longdash", color = "#FF4500", alpha = 0.5, linewidth = 0.5) +
    
    labs(title = paste("Response of", VAR_NAME, "from Sentiment Shock"),
         x = "Months",
         y = VAR_NAME)
  plot_list[[VAR_NAME]] <- plot
}

grid.arrange(grobs = plot_list, ncol = 3, nrow = 3)






plot <- ggplot(subset(results_df, name == 'sent'), 
               aes(x = t)) +
  # lp
  geom_ribbon(aes(ymin = lp_lower, ymax = lp_upper), fill = "gray", alpha = 0.5) +
  geom_line(aes(y = lp_mean), color = "black", linewidth = 0.5) +
  geom_line(aes(y = lp_lower), color = "gray", linewidth = 0.5) +
  geom_line(aes(y = lp_upper),  color = "gray", linewidth = 0.5) +
  # var
  geom_ribbon(aes(ymin = var_lower, ymax = var_upper), fill = "#FF4500", alpha = 0.3) +
  geom_line(aes(y = var_mean), linetype = "longdash", color = "#FF4500", linewidth = 0.5) +
  geom_line(aes(y = var_lower), linetype = "longdash", color = "#FF4500", alpha = 0.5, linewidth = 0.5) +
  geom_line(aes(y = var_upper), linetype = "longdash", color = "#FF4500", alpha = 0.5, linewidth = 0.5) +
  
  labs(title = paste("Response of", VAR_NAME, "from Sentiment Shock"),
       x = "Months",
       y = VAR_NAME)


plot


index <-1
lp_mean <- results_lin$irf_lin_mean[index,  , 1]
var_mean <- var_irf_sent$irf$sent[, index] 

lp_mean
var_mean

var_irf_sent$irf$sent[1, index] 

t <- 2
lp_mean <- results_lin$irf_lin_mean[index, t , 1]
lp_mean
length(lp_mean)
length(var_mean)

#length(name)
#length(index)
