# -*- coding: utf-8 -*-

'''
Monolito de aplicacao de um AG simples para otimizacao da funcao f(x) = x^2
'''

# --------------------------------------------
# --------------------------------------------
# IMPORTACAO DAS BIBLIOTECAS
# --------------------------------------------
# --------------------------------------------

from statistics import mean
from typing import Union, List
import numpy as np
from collections import Counter
import pandas as pd
from pathlib import Path

import ipdb

ipdb.set_trace()

# from collections import Counter
# import numpy as np  # biblioteca para diferentes calculos numericos
# import pandas as pd

# --------------------------------------------
# --------------------------------------------
# DEFINICAO DO PROBLEMA
# --------------------------------------------
# --------------------------------------------

# Definicao do Problema:
# # Dada uma Caixa Preta, contendo 5 interruptores com um sinal de entrada
# # de 0 ou 1 (Desligado ou Ligado), defina a configuracao de interruptores
# # com o maior sinal de saida (recompensa), atraves da maximizacao da fun-
# # cao objetivo f(x) = x**2, onde x pode apresentar qualquer inteiro na
# # intervalo [0, 31].
# #
# # Adaptado de Goldberg (1889)

# --------------------------------------------
# --------------------------------------------
# VARIAVEIS GLOBAIS
# --------------------------------------------
# --------------------------------------------

# Cada variavel global sera representada por letras maiusculas.

# --------------------------------------------
# INDICADORES PARAMETRIZAVEIS
# --------------------------------------------

# tamanho da populacao
N = 5 
# numero de caracteres de cada matriz
NUM_CARACTERISTICAS = 5
# numero de caracteres de cada matriz
PROBABILIDADE_MUTACAO = 0.001
# indicador de uso do gerador de numeroes pseudo aleatorios
# se 1, entao usar gerador; se 0, nao usar;
USAR_RANDOMSTATE = 1
# indicador de uso da populacao gerada como exemplo na monografia ...
# DA COSTA, E. (2022) ou qualquer outra populacao predefinida
# se 1, entao usar populacao predeterminada; se 0, nao usar;
USAR_POPULACAO_INICIAL_PREDETERMINADA = 1
# indicador de uso da matriz nao selecionada para cruzamento no reservatorio
# de cruzamento. Esse caso ocorre quando a populacao tem numero par
# se 1, formar um par para a matriz nao selecionada, senao, ignorar matriz
CONSIDERAR_MATRIZ_SEM_PAR = 1
# numero maximo de geracoes (criterio de parada)
NUM_MAX_GERACOES = 2

# --------------------------------------------
# VARIAVEIS DINAMICAS
# --------------------------------------------

# contador de iteracoes (passagem de geracoes)
CONTADOR_GERACAO = 0
# ---
# populacao inicial gerada monografia DA COSTA, E. (2022)
POPULACAO_INICIAL_PREDEFINIDA = '0101100111101111110001011'
# populacao inicial gerada aleatoriamente
POPULACAO = {}
# ---
# dicionario com os dados dos processos do AG ate a aplicacao dos operadores
BASE_POPULACOES_PRE_OPERADORES = {}
# dicionario com os dados dos processos do AG ate a aplicacao dos operadores
BASE_POPULACOES_POS_OPERADORES = {}

# gerador de numeros pseudo aleatorios.
# necessario para replicacao dos resultados, quando se deseja os mesmos nu-
# meros de saida apos varias execucoes do script.
if (USAR_RANDOMSTATE == 0):
    rnd = np.random.RandomState()
else:
    rnd = np.random.RandomState(15)

# --------------------------------------------
# --------------------------------------------
# FUNCAO OBJETIVO E FUNCAO DE CONVERSAO DE VALORES DE BASE 2 PARA BASE 10
# --------------------------------------------
# --------------------------------------------

def funcao_objetivo(valor_x: Union[int, float]) -> Union[int, float]:
    """Funcao f(x) = x^2. Recebe o valor do alelo decodificado e retorna
    o valor de aptidao.

    Parameters
    ----------
    valor_x : int | float
        Numero decodificado para aplicacao da funcao objetivo.
    
    Return
    ----------
    int | float
        valor de aptidao.
    """
    return valor_x ** 2

def decodificacao_base_dois(matriz: List[str]) -> List[int]:
    """Decodificacao dos valores binarios do sistema numerico de base 2 para 
    inteiros de base 10 ${a_{ij}}^{n-i}$.

    Parameters
    ----------
    matriz : list
        Lista com os valores binarios para decodificacao.
    
    Return
    ----------
    list
        uma lista com os valores binarios decodificados para base 10.
    """
    base = 2
    tamanho_matriz = len(matriz)
    lista_binarios = [_ for _ in range(tamanho_matriz)]
    lista_binarios_decodificados = []
    for id_binario in lista_binarios:
        valor_binario = int(matriz[id_binario])
        valor_base_dez = valor_binario * base ** (tamanho_matriz - 1 - id_binario)
        lista_binarios_decodificados.append(valor_base_dez)
    return lista_binarios_decodificados

# --------------------------------------------
# --------------------------------------------
# 1. CONSTRUCAO DA POPULACAO INICIAL
# --------------------------------------------
# --------------------------------------------

# opcoes de criterio de parada
OPCOES_CRITERIO_PARADA = ['num_maximo_geracoes', 'valor_aptidao_maximo_alcancado', 'valor_aptidao_corrente_menor_que_anterior']
# ---
criterio_parada = OPCOES_CRITERIO_PARADA[1]
# ---
binario_num_max_geracoes = False
binario_valor_mais_aptidao_alcancado = False
binario_valor_aptidao_corrente_menor_que_anterior = False

# iterador de geracoes levando em consideracao os criterios de parada
while not (
    binario_num_max_geracoes if criterio_parada == 'num_maximo_geracoes' else (
        binario_valor_mais_aptidao_alcancado if criterio_parada == 'valor_aptidao_maximo_alcancado' else binario_valor_aptidao_corrente_menor_que_anterior
        )
    ):

    print(f'Inicio: Populacao Inicial/Geracao {CONTADOR_GERACAO+1}' if CONTADOR_GERACAO == 0 else f'Inicio: Geracao {CONTADOR_GERACAO + 1}')

    breakpoint()

    # se base de dados das populacoes estiverem vazias, gera a populacao inicial
    if not BASE_POPULACOES_PRE_OPERADORES and not BASE_POPULACOES_POS_OPERADORES:
        num_lancamentos = N * NUM_CARACTERISTICAS
        lados_moeda = [0, 1]
        probabilidades = [.5, .5]
        # lancamento da moeda para cada caracteristica da populacao
        lista_lancamentos_moeda = rnd.choice(lados_moeda, size=num_lancamentos, p=probabilidades)
        # ---
        # usar populacao inicial predefinida
        if USAR_POPULACAO_INICIAL_PREDETERMINADA == 1:
            lista_caracteres_predefinidos = [int(caractere) for caractere in POPULACAO_INICIAL_PREDEFINIDA]
            for id_caractere, caractere in enumerate(lista_caracteres_predefinidos):
                num_caractere = f'valor_alelo_{id_caractere+1}'
                POPULACAO[num_caractere] = caractere
        # usar populacao inicial aleatoria
        else:
            for id_lancamento, lancamento in enumerate(lista_lancamentos_moeda):
                num_lancamento = f'valor_alelo_{id_lancamento+1}'
                POPULACAO[num_lancamento] = lancamento
    else:
        populacao_interior = BASE_POPULACOES_POS_OPERADORES.get(f'id_geracao_{CONTADOR_GERACAO-1}')
        alelos_populacao = []
        for matriz in populacao_interior:
            matriz_populacao_anteior = populacao_interior.get(matriz)['matriz_apos_operador_mutacao']
            for alelo in matriz_populacao_anteior:
                alelos_populacao.append(alelo)
        for id_alelo, alelo in enumerate(alelos_populacao):
            num_caractere = f'valor_alelo_{id_alelo+1}'
            POPULACAO[num_caractere] = alelo         

    # --------------------------------------------
    # 1.1 AGRUPAMENTO DOS CARACTERES DA POPULACAO (CRIACAO MATRIZES)
    # --------------------------------------------

    # dicionario com as informacoes gerada das matrizes da populacao inicial
    matrizes = {}
    i = 0 # numero de particionamento inicial
    for num_caracteristica in range(NUM_CARACTERISTICAS):
        lista_posicao_alelos_populacao = [locus for locus in range(len(POPULACAO))]
        lista_alelos_populacao = list(POPULACAO.values())
        # ---
        # quebra da populacao por individuo e caracteristicas
        alelos_lista = lista_posicao_alelos_populacao[i:i+NUM_CARACTERISTICAS]
        matriz_lista = lista_alelos_populacao[i:i+NUM_CARACTERISTICAS]
        locus_lista = [locus for locus in range(len(alelos_lista))]
        # ---
        # envio dos dados construidos para um dicionario
        key = f'id_{num_caracteristica+1}'
        matriz_str = ''.join([str(alelo) for alelo in matriz_lista])
        matrizes[key] = {
            'matriz': matriz_str,
            'lista_id_alelo_populacao': alelos_lista,
            'lista_locus': locus_lista,
            'lista_valor_alelo': matriz_lista,
        }
        i += NUM_CARACTERISTICAS

    # --------------------------------------------
    # 1.2 DECODIFICACAO DOS VALORES BINARIOS DAS MATRIZES
    # --------------------------------------------

    for id_matriz, matriz in enumerate(matrizes):
        alelos_int = matrizes.get(matriz)
        alelos_str = [str(alelo) for alelo in alelos_int['lista_valor_alelo']]
        # ---
        # conversao de base 2 para base 10
        alelos_decodificados = (decodificacao_base_dois(alelos_str))
        # conversao de base 2 para base 10 levando em considracao que todos os valores dos alelos sao 1
        # necessario para a definicao do criterio de parada
        alelos_decodificados_maximos = decodificacao_base_dois('1' * len(alelos_str))
        # ---
        # alimentacao dos resultados do dicionario das matrizes
        matrizes.get(matriz).update({'lista_alelos_decodificados': alelos_decodificados})
        matrizes.get(matriz).update({'lista_alelos_decodificados_maximos': alelos_decodificados_maximos})

    # --------------------------------------------
    # 1.2.1 SOMA DOS VALORES DECODIFICADOS
    # --------------------------------------------

    for id_matriz, matriz in enumerate(matrizes):
        lista_alelos_decodificados = matrizes.get(matriz)['lista_alelos_decodificados']
        lista_alelos_decodificados_maximos = matrizes.get(matriz)['lista_alelos_decodificados_maximos']
        # ---
        # alimentacao dos resultados do dicionario das matrizes
        matrizes.get(matriz).update({'valor_de_x': sum(lista_alelos_decodificados)})
        matrizes.get(matriz).update({'max_valor_de_x_possivel': sum(lista_alelos_decodificados_maximos)})

    # --------------------------------------------
    # --------------------------------------------
    # 2 CALCULO DOS VALORES DE APTIDAO
    # --------------------------------------------
    # --------------------------------------------

    for id_matriz, matriz in enumerate(matrizes):
        valor_x = matrizes.get(matriz)['valor_de_x']
        valor_maximo_x = matrizes.get(matriz)['max_valor_de_x_possivel']
        # ---
        valor_aptidao = funcao_objetivo(valor_x)
        valor_maximo_aptidao = funcao_objetivo(valor_maximo_x)
        # ---
        # alimentacao dos resultados do dicionario das matrizes
        matrizes.get(matriz).update({'valor_aptidao': valor_aptidao})
        matrizes.get(matriz).update({'max_valor_aptidao_possivel': valor_maximo_aptidao})

    # --------------------------------------------
    # 2.1 ORDENACAO DAS MATRIZES
    # --------------------------------------------

    dicionario_matrizes_temp = {}
    for matriz in matrizes:
        dicionario_matrizes_temp[matriz] = matrizes.get(matriz)['valor_aptidao']

    matrizes_ordenadas = {}
    for matriz in sorted(dicionario_matrizes_temp, key=dicionario_matrizes_temp.get, reverse=True):
        matrizes_ordenadas[matriz] = matrizes.get(matriz)

    # --------------------------------------------
    # 2.2 CALCULO DOS MINIMOS, MEDIOS, SOMAS E MAXIMOS DAS MATRIZES
    # --------------------------------------------

    lista_ids_matrizes = [id_matriz for id_matriz in matrizes_ordenadas.keys()]

    for matriz in matrizes_ordenadas:
        min_valor_x = min([matrizes_ordenadas.get(_matriz)['valor_de_x'] for _matriz in lista_ids_matrizes])
        media_valor_x = mean([matrizes_ordenadas.get(_matriz)['valor_de_x'] for _matriz in lista_ids_matrizes])
        soma_valor_x = sum([matrizes_ordenadas.get(_matriz)['valor_de_x'] for _matriz in lista_ids_matrizes])
        max_valor_x = max([matrizes_ordenadas.get(_matriz)['valor_de_x'] for _matriz in lista_ids_matrizes])
        # ---
        # alimentacao dos resultados do dicionario das matrizes
        matrizes_ordenadas.get(matriz).update({'min_valor_x': min_valor_x})
        matrizes_ordenadas.get(matriz).update({'media_valor_x': media_valor_x})
        matrizes_ordenadas.get(matriz).update({'soma_valor_x': soma_valor_x})
        matrizes_ordenadas.get(matriz).update({'max_valor_x': max_valor_x})

    for matriz in matrizes_ordenadas:
        min_valor_aptidao = min([matrizes_ordenadas.get(_matriz)['valor_aptidao'] for _matriz in lista_ids_matrizes])
        media_valor_aptidao = mean([matrizes_ordenadas.get(_matriz)['valor_aptidao'] for _matriz in lista_ids_matrizes])
        soma_valor_aptidao = sum([matrizes_ordenadas.get(_matriz)['valor_aptidao'] for _matriz in lista_ids_matrizes])
        max_valor_aptidao = max([matrizes_ordenadas.get(_matriz)['valor_aptidao'] for _matriz in lista_ids_matrizes])
        # ---
        # alimentacao dos resultados do dicionario das matrizes
        matrizes_ordenadas.get(matriz).update({'min_valor_aptidao': min_valor_aptidao})
        matrizes_ordenadas.get(matriz).update({'media_valor_aptidao': media_valor_aptidao})
        matrizes_ordenadas.get(matriz).update({'soma_valor_aptidao': soma_valor_aptidao})
        matrizes_ordenadas.get(matriz).update({'max_valor_aptidao': max_valor_aptidao})

    # --------------------------------------------
    # --------------------------------------------
    # 3. APLICACAO DOS OPERADORES
    # --------------------------------------------
    # --------------------------------------------

    # --------------------------------------------
    # 3.1 REPRODUCAO
    # --------------------------------------------

    # --------------------------------------------
    # 3.1.1 CALCULO DA PROBABILIDADE DE SELECAO
    # --------------------------------------------

    for matriz in matrizes_ordenadas:
        valor_aptidao = matrizes_ordenadas.get(matriz)['valor_aptidao']
        soma_valor_aptidao = matrizes_ordenadas.get(matriz)['soma_valor_aptidao']
        probabilidade_selecao = (valor_aptidao / soma_valor_aptidao)
        probabilidade_selecao_norm = round(probabilidade_selecao * 100, 1)
        # ---
        # alimentacao dos resultados do dicionario das matrizes
        matrizes_ordenadas.get(matriz).update({'probabilidade_selecao': probabilidade_selecao})
        matrizes_ordenadas.get(matriz).update({'probabilidade_selecao_norm': probabilidade_selecao_norm})

    # --------------------------------------------
    # 3.1.2 CALCULO DO NUMERO ESPERADO DE REPRODUCOES
    # --------------------------------------------

    for matriz in matrizes_ordenadas:
        valor_aptidao = matrizes_ordenadas.get(matriz)['valor_aptidao']
        media_valor_aptidao = matrizes_ordenadas.get(matriz)['media_valor_aptidao']
        num_esperado_reproducao = (valor_aptidao / media_valor_aptidao)
        num_esperado_reproducao_norm = round(num_esperado_reproducao, 2)
        # ---
        # alimentacao dos resultados do dicionario das matrizes
        matrizes_ordenadas.get(matriz).update({'numero_esperado_reproducao_norm': num_esperado_reproducao_norm})

    # --------------------------------------------
    # 3.1.3 SORTEIO DAS MATRIZES PARA REPRODUCAO
    # --------------------------------------------

    lista_ids_matrizes = [matriz for matriz in matrizes_ordenadas]
    probabilidades_reproducao = [matrizes_ordenadas.get(matriz)['probabilidade_selecao'] for matriz in matrizes_ordenadas]

    # sorteio das matrizes para reproducao
    matrizes_selecionadas_reproducao = list(rnd.choice(lista_ids_matrizes, size=len(lista_ids_matrizes), p=probabilidades_reproducao))

    for matriz in matrizes_ordenadas:
        selecao_roleta = dict(Counter(matrizes_selecionadas_reproducao)).get(matriz)
        # ---
        # alimentacao dos resultados do dicionario das matrizes
        matrizes_ordenadas[matriz].update({'numero_de_reproducoes': 0 if not selecao_roleta else selecao_roleta})

    # --------------------------------------------
    # 3.1.4 CALCULO DOS MINIMOS, MEDIOS, SOMAS E MAXIMOS DAS MATRIZES
    # --------------------------------------------

    lista_ids_matrizes = [id_matriz for id_matriz in matrizes_ordenadas.keys()]

    for matriz in matrizes_ordenadas:
        min_probabilidade_selecao = min([matrizes_ordenadas.get(_matriz)['probabilidade_selecao'] for _matriz in lista_ids_matrizes])
        media_probabilidade_selecao = mean([matrizes_ordenadas.get(_matriz)['probabilidade_selecao'] for _matriz in lista_ids_matrizes])
        soma_probabilidade_selecao = sum([matrizes_ordenadas.get(_matriz)['probabilidade_selecao'] for _matriz in lista_ids_matrizes])
        max_probabilidade_selecao = max([matrizes_ordenadas.get(_matriz)['probabilidade_selecao'] for _matriz in lista_ids_matrizes])
        # ---
        # alimentacao dos resultados do dicionario das matrizes
        matrizes_ordenadas.get(matriz).update({'min_probabilidade_selecao': min_probabilidade_selecao})
        matrizes_ordenadas.get(matriz).update({'media_probabilidade_selecao': media_probabilidade_selecao})
        matrizes_ordenadas.get(matriz).update({'soma_probabilidade_selecao': soma_probabilidade_selecao})
        matrizes_ordenadas.get(matriz).update({'max_probabilidade_selecao': max_probabilidade_selecao})

    for matriz in matrizes_ordenadas:
        min_numero_esperado_reproducao_norm = min([matrizes_ordenadas.get(_matriz)['numero_esperado_reproducao_norm'] for _matriz in lista_ids_matrizes])
        media_numero_esperado_reproducao_norm = mean([matrizes_ordenadas.get(_matriz)['numero_esperado_reproducao_norm'] for _matriz in lista_ids_matrizes])
        soma_numero_esperado_reproducao_norm = sum([matrizes_ordenadas.get(_matriz)['numero_esperado_reproducao_norm'] for _matriz in lista_ids_matrizes])
        max_numero_esperado_reproducao_norm = max([matrizes_ordenadas.get(_matriz)['numero_esperado_reproducao_norm'] for _matriz in lista_ids_matrizes])
        # ---
        # alimentacao dos resultados do dicionario das matrizes
        matrizes_ordenadas.get(matriz).update({'min_numero_esperado_reproducao_norm': min_numero_esperado_reproducao_norm})
        matrizes_ordenadas.get(matriz).update({'media_numero_esperado_reproducao_norm': media_numero_esperado_reproducao_norm})
        matrizes_ordenadas.get(matriz).update({'soma_numero_esperado_reproducao_norm': soma_numero_esperado_reproducao_norm})
        matrizes_ordenadas.get(matriz).update({'max_numero_esperado_reproducao_norm': max_numero_esperado_reproducao_norm})

    for matriz in matrizes_ordenadas:
        min_numero_de_reproducoes = min([matrizes_ordenadas.get(_matriz)['numero_de_reproducoes'] for _matriz in lista_ids_matrizes])
        media_numero_de_reproducoes = mean([matrizes_ordenadas.get(_matriz)['numero_de_reproducoes'] for _matriz in lista_ids_matrizes])
        soma_numero_de_reproducoes = sum([matrizes_ordenadas.get(_matriz)['numero_de_reproducoes'] for _matriz in lista_ids_matrizes])
        max_numero_de_reproducoes = max([matrizes_ordenadas.get(_matriz)['numero_de_reproducoes'] for _matriz in lista_ids_matrizes])
        # ---
        # alimentacao dos resultados do dicionario das matrizes
        matrizes_ordenadas.get(matriz).update({'min_numero_de_reproducoes': min_numero_de_reproducoes})
        matrizes_ordenadas.get(matriz).update({'media_numero_de_reproducoes': media_numero_de_reproducoes})
        matrizes_ordenadas.get(matriz).update({'soma_numero_de_reproducoes': soma_numero_de_reproducoes})
        matrizes_ordenadas.get(matriz).update({'max_numero_de_reproducoes': max_numero_de_reproducoes})

    # --------------------------------------------
    # 3.2 CRUZAMENTO
    # --------------------------------------------

    # --------------------------------------------
    # 3.2.1 FORMACAO DOS PARES NO RESERVATORIO DE CRUZAMENTO
    # --------------------------------------------

    # construcao de nova lista com as matrizes selecionadas para reproducao
    lista_matrizes_selecionadas_reproducao = []
    for matriz in matrizes_ordenadas:
        num_reproducao = matrizes_ordenadas.get(matriz)['numero_de_reproducoes']
        if num_reproducao != 0:
            lista_matrizes_reproducao = [matriz] * num_reproducao
            for matriz_reproducao in lista_matrizes_reproducao:
                lista_matrizes_selecionadas_reproducao.append(matriz_reproducao)

    # construcao do reservatorio de cruzamento com as matrizes reproduzidas
    matrizes_reservatorio_cruzamento = {}
    for id_matriz, matriz in enumerate(lista_matrizes_selecionadas_reproducao):
        id_matriz_reservatorio = f'id_matriz_reservatorio_acasalamento_{id_matriz}'
        # envio das informacoes das matrizes reproduzidas para um novo dicionario
        matrizes_reservatorio_cruzamento[id_matriz_reservatorio] = {
            'id_matriz': matriz,
            'matriz': matrizes_ordenadas.get(matriz)['matriz'],
            'lista_valor_alelo': matrizes_ordenadas.get(matriz)['lista_valor_alelo']
        }

    # lista contendo as matrizes sorteadas, de forma excludente, para cruzamento
    matrizes_selecionadas_cruzamento = []
    for matriz in matrizes_reservatorio_cruzamento:
        lista_matrizes = [
            id_matriz for id_matriz
            in matrizes_reservatorio_cruzamento
            if id_matriz != matriz
            and id_matriz not in matrizes_selecionadas_cruzamento
        ]
        # sorteio das matrizes e construcao da lista de pares
        if lista_matrizes:
            probabilidades = [1 / len(lista_matrizes) for _ in lista_matrizes]
            matriz_sorteada = rnd.choice(lista_matrizes, size=1, p=probabilidades)[0]
            matrizes_selecionadas_cruzamento.append(matriz_sorteada)

    # novo dicionario com as matrizes do reservatorio de cruzamento formadas
    pares_reservatorio_cruzamento = {}
    i = 0
    for id_par in range(int(len(matrizes_selecionadas_cruzamento) / 2)):
        par = matrizes_selecionadas_cruzamento[i:i+2]
        pares_reservatorio_cruzamento[f'par_{id_par}'] = {
            'ids_matrizes_pares': par,
            'par_com_matriz_nao_selecionada': False
        }
        i += 2

    # no caso do numero de matrizes no reservatorio de cruzamento ser imprar,
    # uma matriz ficara de fora do cruzamento. Para este caso, caso seja
    # necessario considera-la, esta matriz nao selecionada formara par com
    # uma matriz da lista de matrizes que ja possuem par
    if CONSIDERAR_MATRIZ_SEM_PAR == 1:
        matrizes_selecionadas = []
        for par in pares_reservatorio_cruzamento:
            par_matrizes = pares_reservatorio_cruzamento.get(par)['ids_matrizes_pares']
            for matriz in par_matrizes:
                matrizes_selecionadas.append(matriz)
        # ---
        matriz_nao_selecionada = list(set(matrizes_reservatorio_cruzamento) - set(matrizes_selecionadas))
        # ---
        if matriz_nao_selecionada:
            probabilidades = [1 / len(matrizes_selecionadas) for _ in matrizes_selecionadas]
            # sorteio de matriz dentre as que ja foram sorteadas anteriormente
            matriz_sorteada = rnd.choice(matrizes_selecionadas, size=1, p=probabilidades)[0]
            num_par = len(pares_reservatorio_cruzamento)
            # alimentacao dos resultados do dicionario das matrizes
            pares_reservatorio_cruzamento.update(
                {
                    f'par_{num_par}': 
                        {
                            'ids_matrizes_pares': [matriz_nao_selecionada[0], matriz_sorteada],
                            'par_com_matriz_nao_selecionada': True
                            }
                    }
                    )

    # alimentacao do dicionario de matrizes pares
    for par in pares_reservatorio_cruzamento:
        par_matrizes = pares_reservatorio_cruzamento.get(par)['ids_matrizes_pares']
        id_matriz_1, id_matriz_2 = par_matrizes[0], par_matrizes[1]
        # ---
        matriz_1 = matrizes_reservatorio_cruzamento.get(id_matriz_1)['matriz']
        matriz_2 = matrizes_reservatorio_cruzamento.get(id_matriz_2)['matriz']
        # ---
        lista_valor_alelo_1 = matrizes_reservatorio_cruzamento.get(id_matriz_1)['lista_valor_alelo']
        lista_valor_alelo_2 = matrizes_reservatorio_cruzamento.get(id_matriz_2)['lista_valor_alelo']
        # alimentacao dos resultados do dicionario das matrizes
        pares_reservatorio_cruzamento.get(par).update({
            'matrizes_pares': [matriz_1, matriz_2],
            'lista_valor_alelo_matrizes_pares': [lista_valor_alelo_1, lista_valor_alelo_2]
        })

    # --------------------------------------------
    # 3.2.2 SORTEIO DOS PONTOS DE CRUZAMENTO
    # --------------------------------------------

    # escolha aleatoria do ponto de corte para cruzamento entre os pares
    for par in pares_reservatorio_cruzamento:
        matrizes_pares = pares_reservatorio_cruzamento.get(par)
        matriz_1, matriz_2 = matrizes_pares['matrizes_pares'][0], matrizes_pares['matrizes_pares'][1]
        lista_valor_alelo_1, lista_valor_alelo_2 = matrizes_pares['lista_valor_alelo_matrizes_pares'][0], matrizes_pares['lista_valor_alelo_matrizes_pares'][1]
        tamanho_matriz = int((len(matriz_1) + len(matriz_2)) / 2) - 1
        # sorteio do ponto de cruzamento para troca de informacoes
        pontos_corte = [ponto+1 for ponto in range(tamanho_matriz)]
        probabilidades = [1 / tamanho_matriz for _ in range(tamanho_matriz)]
        ponto_sorteado = rnd.choice(pontos_corte, size=1, p=probabilidades)[0]
        # ---
        # inclusao dos pontos de cruzamento e ponto de cruzamento sorteado
        lista_valor_alelo_1, lista_valor_alelo_2 = [str(alelo) for alelo in lista_valor_alelo_1], [str(alelo) for alelo in lista_valor_alelo_2]
        lista_alelo_ponto_corte_1 = '.'.join(lista_valor_alelo_1[:ponto_sorteado]) + '|' + '.'.join(lista_valor_alelo_1[ponto_sorteado:])
        lista_alelo_ponto_corte_2 = '.'.join(lista_valor_alelo_2[:ponto_sorteado]) + '|' + '.'.join(lista_valor_alelo_2[ponto_sorteado:])
        # alimentacao dos resultados do dicionario das matrizes
        pares_reservatorio_cruzamento.get(par).update({
            'ponto_corte_cruzamento': ponto_sorteado,
            'ponto_corte_cruzamento_matrizes_pares': [lista_alelo_ponto_corte_1, lista_alelo_ponto_corte_2]
        })

    # --------------------------------------------
    # 3.2.3 CRIACAO DAS NOVAS MATRIZES
    # --------------------------------------------

    # troca de informaoess entre as matrizes pares
    for par in pares_reservatorio_cruzamento:
        matrizes_pares = pares_reservatorio_cruzamento.get(par)
        ponto_cruzamento = matrizes_pares['ponto_corte_cruzamento']
        matriz_1, matriz_2 = matrizes_pares['matrizes_pares'][0], matrizes_pares['matrizes_pares'][1]
        # cruzamento das matrizes
        matriz_cruzada_1, matriz_cruzada_2 = matriz_1[:ponto_cruzamento] + matriz_2[ponto_cruzamento:], matriz_2[:ponto_cruzamento] + matriz_1[ponto_cruzamento:]
        # alimentacao dos resultados do dicionario das matrizes
        pares_reservatorio_cruzamento.get(par).update({
            'matrizes_pares_cruzadas': [matriz_cruzada_1, matriz_cruzada_2]
        })

    # ---

    # criacao de lista temporaria de dicionarios com as infos das matrizes pares
    lista_dict_temp = []
    for par in pares_reservatorio_cruzamento:
        matrizes_pares = pares_reservatorio_cruzamento.get(par)['ids_matrizes_pares']
        for matriz in matrizes_pares:
            infos_dict = {**{'id_par': par}, **pares_reservatorio_cruzamento.get(par)}
            lista_dict_temp.append(infos_dict)

    # dicionario com as informacoes das matrizes apos o cruzamento
    matrizes_cruzadas_reservatorio_cruzamento = {}
    for id_matriz_cruzada, matriz_cruzada in enumerate(lista_dict_temp):
        id_matriz_cruzada_reservatorio = f'id_matriz_cruzada_reservatorio_{id_matriz_cruzada}'
        # novo id da matriz apos o cruzamento
        matrizes_cruzadas_reservatorio_cruzamento[id_matriz_cruzada_reservatorio] = {
            'informacao_pares': matriz_cruzada
        }
        # alimentacao do dicionario com as informacoes da matriz respectiva
        # se id_matriz_par for par, entao pegar o primeiro valor das listas
        # se id_matriz_par for impar, entao pegar o segundo valor das listas
        if (id_matriz_cruzada % 2) == 0: # par
            matrizes_cruzadas_reservatorio_cruzamento.get(id_matriz_cruzada_reservatorio).update({
                'id_matriz_pai': matriz_cruzada.get('ids_matrizes_pares')[0],
                'matriz_pai': matriz_cruzada.get('matrizes_pares')[0],
                'lista_valor_alelo_matriz_pai': matriz_cruzada.get('lista_valor_alelo_matrizes_pares')[0],
                'ponto_cruzamento': matriz_cruzada.get('ponto_corte_cruzamento'),
                'ponto_corte_cruzamento_matriz_pai': matriz_cruzada.get('ponto_corte_cruzamento_matrizes_pares')[0],
                'matriz_filha': matriz_cruzada.get('matrizes_pares_cruzadas')[0],
                'lista_valor_alelo_matriz_filha': [alelo for alelo in matriz_cruzada.get('matrizes_pares_cruzadas')[0]]
            })
        else: # impar
            matrizes_cruzadas_reservatorio_cruzamento.get(id_matriz_cruzada_reservatorio).update({
                'id_matriz_pai': matriz_cruzada.get('ids_matrizes_pares')[1],
                'matriz_pai': matriz_cruzada.get('matrizes_pares')[1],
                'lista_valor_alelo_matriz_pai': matriz_cruzada.get('lista_valor_alelo_matrizes_pares')[1],
                'ponto_cruzamento': matriz_cruzada.get('ponto_corte_cruzamento'),
                'ponto_corte_cruzamento_matriz_pai': matriz_cruzada.get('ponto_corte_cruzamento_matrizes_pares')[1],
                'matriz_filha': matriz_cruzada.get('matrizes_pares_cruzadas')[1],
                'lista_valor_alelo_matriz_filha': [alelo for alelo in matriz_cruzada.get('matrizes_pares_cruzadas')[1]]
            })

    # --------------------------------------------
    # 3.3 MUTACAO
    # --------------------------------------------

    # aplicacao do operador de mutacao
    for matriz in matrizes_cruzadas_reservatorio_cruzamento:
        alelos_matriz_filha = matrizes_cruzadas_reservatorio_cruzamento.get(matriz)['lista_valor_alelo_matriz_filha']
        # lista com os alelos apos o operador de mutacao
        alelos_apos_mutacao = []
        for alelo in alelos_matriz_filha:
            if int(alelo) == 0:
                alelos = [0, 1] # possibilidades
                probabilidades = [1-PROBABILIDADE_MUTACAO, PROBABILIDADE_MUTACAO] # probabilidade de nao mutacao e mutacao
                alelo_apos_operador = rnd.choice(alelos, size=1, p=probabilidades)[0]
                alelos_apos_mutacao.append(alelo_apos_operador)
            else:
                alelos = [1, 0] # possibilidades
                probabilidades = [1-PROBABILIDADE_MUTACAO, PROBABILIDADE_MUTACAO] # probabilidade de nao mutacao e mutacao
                alelo_apos_operador = rnd.choice(alelos, size=1, p=probabilidades)[0]
                alelos_apos_mutacao.append(alelo_apos_operador)
        # ---
        alelos_apos_mutacao_str = [str(alelo) for alelo in alelos_apos_mutacao]
        alelos_matriz_filha = matrizes_cruzadas_reservatorio_cruzamento.get(matriz)['lista_valor_alelo_matriz_filha']
        mutacoes_esperadas = N * NUM_CARACTERISTICAS * PROBABILIDADE_MUTACAO
        # alimentacao dos resultados do dicionario das matrizes
        matrizes_cruzadas_reservatorio_cruzamento.get(matriz).update({
            'matriz_apos_operador_mutacao': ''.join(alelos_apos_mutacao_str),
            'lista_valor_alelo_matriz_apos_operador_mutacao': alelos_apos_mutacao_str,
            'houve_mutacao': 0 if alelos_apos_mutacao_str == alelos_matriz_filha else 1, # 0 se nao houve mutacao; 1 se houve;
            'num_esperado_mutacoes': mutacoes_esperadas
        })

    # --------------------------------------------
    # --------------------------------------------
    # ALIMENTACAO DAS BASES FINAIS
    # --------------------------------------------
    # --------------------------------------------

    id_geracao = f'id_geracao_{CONTADOR_GERACAO}'

    BASE_POPULACOES_PRE_OPERADORES[id_geracao] = matrizes_ordenadas
    BASE_POPULACOES_POS_OPERADORES[id_geracao] = matrizes_cruzadas_reservatorio_cruzamento

    print(f'Fim: Populacao Inicial/Geracao {CONTADOR_GERACAO+1}' if CONTADOR_GERACAO == 0 else f'Fim: Geracao {CONTADOR_GERACAO+1}')

    # --------------------------------------------
    # --------------------------------------------
    # ALIMENTACAO DOS BINARIOS DOS CRITERIOS DE PARADA
    # --------------------------------------------
    # --------------------------------------------

    # binario_num_max_geracoes
    if (CONTADOR_GERACAO+1 == NUM_MAX_GERACOES):
        binario_num_max_geracoes = True

    # binario_valor_mais_aptidao_alcancado
    for matriz in matrizes_ordenadas:
        valor_aptidao = matrizes_ordenadas.get(matriz)['valor_aptidao']
        max_valor_aptidao_possivel = matrizes_ordenadas.get(matriz)['max_valor_aptidao_possivel']
        if (valor_aptidao == max_valor_aptidao_possivel):
            binario_valor_mais_aptidao_alcancado = True
    
    # binario_valor_aptidao_corrente_menor_que_anterior
    if CONTADOR_GERACAO >= 1:
        soma_valor_aptidao_geracao_corrente = []
        soma_valor_aptidao_geracao_anterior = []
        for geracao in BASE_POPULACOES_PRE_OPERADORES:
            geracao_corrente = BASE_POPULACOES_PRE_OPERADORES.get(f'id_geracao_{CONTADOR_GERACAO}')
            geracao_anterior = BASE_POPULACOES_PRE_OPERADORES.get(f'id_geracao_{CONTADOR_GERACAO-1}')
            lista_soma_valor_aptidao_geracao_corrente = []
            lista_soma_valor_aptidao_geracao_anterior = []
            for matriz in geracao_corrente:
                lista_soma_valor_aptidao_geracao_corrente.append(geracao_corrente.get(matriz)['soma_valor_aptidao'])
            for matriz in geracao_anterior:
                lista_soma_valor_aptidao_geracao_anterior.append(geracao_anterior.get(matriz)['soma_valor_aptidao'])
            soma_valor_aptidao_geracao_corrente.append(mean(lista_soma_valor_aptidao_geracao_corrente))
            soma_valor_aptidao_geracao_anterior.append(mean(lista_soma_valor_aptidao_geracao_anterior))
        # ---
        if (soma_valor_aptidao_geracao_corrente < soma_valor_aptidao_geracao_anterior):
            binario_valor_aptidao_corrente_menor_que_anterior = True

    CONTADOR_GERACAO += 1

# --------------------------------------------
# --------------------------------------------
# SALVAMENTO DAS BASES FINAIS (JSON)
# --------------------------------------------
# --------------------------------------------

nome_arquivos = 'informacoes_ag_simples'

try:
    Path('data').mkdir(exist_ok=False)
    pd.DataFrame(BASE_POPULACOES_PRE_OPERADORES).to_json(f'data/{nome_arquivos}_pre_operadores.json', indent=4)
    pd.DataFrame(BASE_POPULACOES_POS_OPERADORES).to_json(f'data/{nome_arquivos}_pos_operadores.json', indent=4)
except OSError:
    print('Nao foi possivel salvar os arquivos.')