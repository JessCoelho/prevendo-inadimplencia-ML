#--------------------------------------------------------------------------------
#        Microsoft Power BI para Data Science -  Data Science Academy
#
#        Prevendo a Inadimplência de Clientes com Machine Learning(ML) 
#--------------------------------------------------------------------------------

# Definindo a pasta de trabalho
setwd("D:/Jessica/Documents/POWER BI PARA DATA SCIENCE - DSA/Cap15")
getwd()

#Instalando os pacotes para o projeto
install.packages("Amelia")         # contém funções para tratar valores ausentes
install.packages("caret")          # permite construir modelos de ML e processamento de dados
install.packages("ggplot2")        # construção de dados
install.packages("dplyr")          # tratar e manipular dados
install.packages("reshape")        # modificar o formato de dados
install.packages("randomForest")   # modelo de machine learning no R
install.packages("e1071")          # modelo de machine learning no R

# Carregando os pacotes
library(Amelia)
library(caret)
library(ggplot2)
library(dplyr)
library(reshape)
library(randomForest)
library(e1071)

# Carregando o dataset
# Fonte: https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients
dados_clientes <- read.csv("dados/dataset.csv")

# Visualizando os dados e sua estrutura
View(dados_clientes)
dim(dados_clientes)
str(dados_clientes)
summary(dados_clientes)

# --------------- Análise Exploratória, Limpeza e Transformação ---------------

# Removendo a primeira coluna ID
dados_clientes$ID <- NULL
dim(dados_clientes)
View(dados_clientes)

# Renomeando a coluna de classe(variável alvo)
colnames(dados_clientes)
colnames(dados_clientes)[24] <- "Inadimplente"
colnames(dados_clientes)
View(dados_clientes)

# Verificando valores ausentes e removendo do dataset
sapply(dados_clientes, function(x) sum(is.na(x))) #retorna a quantidade de valores ausentes em cada coluna
?missmap # cria gráfico para apresentar a porcentagem de valores ausentes e observados
missmap(dados_clientes, main = "Valores Missing Observados")
dados_clientes <- na.omit(dados_clientes) # remove ou omite valores ausentes no dataset


# Convertendo os atributos gênero, escolaridade, estado civil e idade para fatores (categorias)
# Renomeando colunas categórias
colnames(dados_clientes)
colnames(dados_clientes)[2]<- "Genero"
colnames(dados_clientes)[3]<- "Escolaridade"
colnames(dados_clientes)[4]<- "Estado_Civil"
colnames(dados_clientes)[5]<- "Idade"
colnames(dados_clientes)
View(dados_clientes)

# Genero
View(dados_clientes$Genero)
str(dados_clientes$Genero)
summary(dados_clientes$Genero)
?cut # converter variável númerica para o tipo fator(categórica no R) e converte o valor da variável
dados_clientes$Genero <- cut(dados_clientes$Genero,
                             c(0,1,2),
                             labels = c("Masculino",
                                        "Feminino"))
View(dados_clientes$Genero)
str(dados_clientes$Genero)

# Escolaridade
str(dados_clientes$Escolaridade)
summary(dados_clientes$Escolaridade)
dados_clientes$Escolaridade <- cut(dados_clientes$Escolaridade,
                                   c(0,1,2,3,4),
                                   labels = c("Pos Graduado",
                                              "Graduado",
                                              "Ensino Medio",
                                              "Outros"))
View(dados_clientes$Escolaridade)
str(dados_clientes$Escolaridade)
summary(dados_clientes$Escolaridade)

# Estado Civil
str(dados_clientes$Estado_Civil)
summary(dados_clientes$Estado_Civil)
dados_clientes$Estado_Civil <- cut(dados_clientes$Estado_Civil,
                                   c(-1,0,1,2,3),
                                   labels = c("Desconhecido",
                                              "Casado",
                                              "Solteiro",
                                              "Outro"))
View(dados_clientes$Estado_Civil)
str(dados_clientes$Estado_Civil)
summary(dados_clientes$Estado_Civil)

# Convertendo a variável para o tipo fator com faixa etária
str(dados_clientes$Idade)
summary(dados_clientes$Idade)
hist(dados_clientes$Idade)
dados_clientes$Idade <- cut(dados_clientes$Idade,
                            c(0,30,50,100),
                            labels = c("Jovem",
                                       "Adulto",
                                       "Idoso"))
View(dados_clientes$Idade)
str(dados_clientes$Idade)
summary(dados_clientes$Idade)


# Convertendo a variável que indica pagamentos para o tipo fator

dados_clientes$PAY_0 <- as.factor(dados_clientes$PAY_0)
dados_clientes$PAY_2 <- as.factor(dados_clientes$PAY_2)
dados_clientes$PAY_3 <- as.factor(dados_clientes$PAY_3)
dados_clientes$PAY_4 <- as.factor(dados_clientes$PAY_4)
dados_clientes$PAY_5 <- as.factor(dados_clientes$PAY_5)
dados_clientes$PAY_6 <- as.factor(dados_clientes$PAY_6)
# com a função as.factor, é convertido o valor da varíavel para o tipo fator, porém o valor da varíavel não é alterado

# Dataset após as conversões
str(dados_clientes)
sapply(dados_clientes, function(x) sum(is.na(x)))
missmap(dados_clientes, main = "Valores Missing Observados")
dados_clientes <- na.omit(dados_clientes)
missmap(dados_clientes, main = "Valores Missing Observados")
dim(dados_clientes)

# Alterando a variável dependente para o tipo fator
str(dados_clientes$Inadimplente)
colnames(dados_clientes)
dados_clientes$Inadimplente <- as.factor(dados_clientes$Inadimplente)
str(dados_clientes$Inadimplente)
View(dados_clientes)

# Total de inadimplentes versus não-inadimplentes
table(dados_clientes$Inadimplente)

# Vejamos as porcentagens entre as classes
prop.table(table(dados_clientes$Inadimplente))

# Plot da distribuição usando ggplot2
qplot(Inadimplente, data = dados_clientes, geom = "bar")+ 
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

# Set seed - usado para pesquisas para gerar números aleatórios em R
?set.seed
set.seed(12345)

# Técnica de Amostragem Estratificada
# Seleciona as linhas de acordo com a variável inadimplente como strata
?createDataPartition # cria um split dos dados(divisão) para treinamento
indice <- createDataPartition(dados_clientes$Inadimplente, p = 0.75, list = FALSE )
# p = relaciona a porcentagem de divisão dos dados, list = FALSE retorna uma matriz de dados e não uma lista
dim(indice)

# Definimos os dados de treinamento como subconjunto do conjuntos de dados original
#com números de indice e linha (conforme identificado acima) e todas as colunas
dados_treino <- dados_clientes[indice,] #linhas(indice), colunas(0)
table(dados_treino$Inadimplente)

# Número de registros no dataset de treinamento
dim(dados_treino)

# Comparamos as porcentagens entre as classes de treinamento e dados originais
compara_dados <- cbind(prop.table(table(dados_treino$Inadimplente)),
                       prop.table(table(dados_clientes$Inadimplente)))
#cbind liga as colunas por proporção para comparação
colnames(compara_dados) <- c("Treinamento", "Original")
compara_dados

#Melt Data - Converte colunas em linhas
?reshape2::melt
melt_compara_dados <- melt(compara_dados)
melt_compara_dados

# Plot para ver a distribuição do treinamento vs original
ggplot(melt_compara_dados, aes(x = X1, y = value))+ 
  geom_bar(aes(fill = X2), stat = "identity", position = "dodge")+
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

# Tudo o que não está no dataset de treinamento está no dataset de teste.
dados_teste <- dados_clientes[-indice,] # -indice , filtra o dataset orginal, menos oq tiver nos dados de treino
dim(dados_teste)
dim(dados_treino)


#-------------------------- Modelos de Machine Learning -------------------------------------

## Construindo a primeira versão do modelo
?randomForest
View(dados_treino)
modelo_v1 <- randomForest(Inadimplente ~ . , data = dados_treino) 
# para o modelo randomForest, é necessário a váriavel alvo(Inadimplente), ~(da formula), .(todas as variáveis preditoras),
#data(o dataset de treino)
modelo_v1

#Avaliando o modelo
plot(modelo_v1)

# Previsões com dados de teste
previsoes_v1 <- predict(modelo_v1, dados_teste)

# Confusion Matrix
?caret::confusionMatrix
cm_v1 <- caret::confusionMatrix(previsoes_v1, dados_teste$Inadimplente, positive = "1")
cm_v1

# Calculando Precision, Recall, F1-score, métricas de avalição do modelo preditivo
y  <- dados_teste$Inadimplente
y_pred_v1 <- previsoes_v1

precision <- posPredValue(y_pred_v1, y)
precision

recall <- sensitivity(y_pred_v1, y)
recall

F1 <- (2* precision * recall)/(precision + recall)
F1

# Balanceamento de classe
install.packages("DMwR")
library(DMwR)
?SMOTE

# Aplicando o SMOTE - SMOTE: Synthetic Minority Over-Sampling Technique
table(dados_treino$Inadimplente)
prop.table(table(dados_treino$Inadimplente))
set.seed(9560)
dados_treino_bal <- SMOTE(Inadimplente ~ . , data = dados_treino)
table(dados_treino_bal$Inadimplente)
prop.table(table(dados_treino_bal$Inadimplente))

## Construindo a segunda versão do modelo(Com dados de treino balanceados)
modelo_v2 <- randomForest(Inadimplente ~ . ,data = dados_treino_bal)
modelo_v2

#Avaliando o modelo
plot(modelo_v2)

# Previsões com dados de teste
previsoes_v2 <- predict(modelo_v2, dados_teste)

# Confusion Matrix
?caret::confusionMatrix
cm_v2 <- caret::confusionMatrix(previsoes_v2, dados_teste$Inadimplente, positive = "1")
cm_v2

# Calculando Precision, Recall, F1-score, métricas de avalição do modelo preditivo
y  <- dados_teste$Inadimplente
y_pred_v2 <- previsoes_v2

precision <- posPredValue(y_pred_v2, y)
precision

recall <- sensitivity(y_pred_v2, y)
recall

F1 <- (2* precision * recall)/(precision + recall)
F1

# Importância das variáveis preditoras para as previsões
View(dados_treino_bal)
varImpPlot(modelo_v2)

# Obtendo as variáveis mais importantes
imp_var <- importance(modelo_v2)
varImportance <- data.frame(Variables = row.names(imp_var), 
                            Importance = round(imp_var[,'MeanDecreaseGini'],2))

# Criando o rank de variáveis baseado na importância
rankImportance <- varImportance %>% mutate(Rank = paste0('#', dense_rank(desc(Importance))))

# Usando ggplot2 para visualizar a importância relatica das variáveis
ggplot(rankImportance,aes(x = reorder(Variables, Importance),
                          y = Importance,
                          fill = Importance)) + 
                          geom_bar(stat = 'identity') + 
                          geom_text(aes(x = Variables, y = 0.5, label = Rank),
                                    hjust = 0, vjust = 0.55, size = 4,
                                    colour = 'red') + labs(x = 'Variables') + coord_flip()

## Construindo a terceira versão do modelo
colnames(dados_treino_bal)
modelo_v3 <- randomForest(Inadimplente ~ PAY_0 + PAY_2 + PAY_3 + PAY_5 + PAY_AMT1 +
                          PAY_AMT2 + BILL_AMT1, data = dados_treino_bal)
modelo_v3

# Avaliando o modelo
plot(modelo_v3)

# Previsões com dados de teste
previsoes_v3 <- predict(modelo_v3, dados_teste)

# Confusion Matrix
?caret::confusionMatrix
cm_v3 <- caret::confusionMatrix(previsoes_v3, dados_teste$Inadimplente, positive = "1")
cm_v3

# Calculando Precision, Recall, F1-score, métricas de avalição do modelo preditivo
y  <- dados_teste$Inadimplente
y_pred_v3 <- previsoes_v3

precision <- posPredValue(y_pred_v3, y)
precision

recall <- sensitivity(y_pred_v3, y)
recall

F1 <- (2* precision * recall)/(precision + recall)
F1

# Salvando os modelos em disco
saveRDS(modelo_v1, file = "modelo/modelo_v1.rds")
saveRDS(modelo_v2, file = "modelo/modelo_v2.rds")
saveRDS(modelo_v3, file = "modelo/modelo_v3.rds")

# Carregando o modelo
modelo_final <- readRDS("modelo/modelo_v3.rds")

#-------------------------- Previsões com o Modelo Treinado -------------------------------------

# Previsões com novos dados de 3 clientes
# Dados dos clientes
PAY_0 <- c(0, 0, 0)
PAY_2 <- c(0, 0, 0)
PAY_3 <- c(1, 0, 0)
PAY_5 <- c(0, 0, 0)
PAY_AMT1 <- c(1100, 1000, 1200)
PAY_AMT2 <- c(1500, 1300, 1150)
BILL_AMT1 <- c(350, 420, 280)

# Concatena em um dataframe
novos_clientes <- data.frame(PAY_0, PAY_2, PAY_3, PAY_5,PAY_AMT1, PAY_AMT2, BILL_AMT1)
View(novos_clientes)

# Previsões
previsões_novos_clientes <- predict(modelo_final, novos_clientes)

# Checando os tipos de dados
str(dados_treino_bal)
str(novos_clientes)

# Convertendo os tipos de dados
novos_clientes$PAY_0 <- factor(novos_clientes$PAY_0, levels = levels(dados_treino_bal$PAY_0))
novos_clientes$PAY_2 <- factor(novos_clientes$PAY_2, levels = levels(dados_treino_bal$PAY_2))
novos_clientes$PAY_3 <- factor(novos_clientes$PAY_3, levels = levels(dados_treino_bal$PAY_3))
novos_clientes$PAY_5 <- factor(novos_clientes$PAY_0, levels = levels(dados_treino_bal$PAY_5))
str(novos_clientes)

# Realizando novamente a previsão
previsões_novos_clientes <- predict(modelo_final, novos_clientes)
View(previsões_novos_clientes)