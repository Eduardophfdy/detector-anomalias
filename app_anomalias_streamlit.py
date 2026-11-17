from io import BytesIO

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


class AnalisadorAnomalias:
    """Detecta e agrupa anomalias em colunas numericas de um DataFrame."""

    def __init__(self, dados, coluna_referencia, sensibilidade=2.0, max_gap=1):
        self.df = dados.reset_index(drop=True).copy()
        self.coluna_referencia = coluna_referencia
        self.sensibilidade = float(sensibilidade)
        self.max_gap = int(max_gap)
        self.resultados = {}
        self.pontos_anomalos = {}

        if coluna_referencia not in self.df.columns:
            raise ValueError(f"A coluna de referencia '{coluna_referencia}' nao existe.")
        if self.sensibilidade <= 0:
            raise ValueError("A sensibilidade deve ser maior que zero.")
        if self.max_gap < 0:
            raise ValueError("max_gap deve ser um inteiro nao negativo.")

    def agrupar_indices(self, indices):
        """Agrupa indices anomalos separados por, no maximo, max_gap linhas normais."""
        grupos = []
        grupo_atual = []

        for indice in indices:
            if not grupo_atual or indice - grupo_atual[-1] <= self.max_gap + 1:
                grupo_atual.append(indice)
            else:
                grupos.append(grupo_atual)
                grupo_atual = [indice]

        if grupo_atual:
            grupos.append(grupo_atual)

        return grupos

    def detectar_parametro(self, nome):
        """Detecta anomalias em uma coluna e retorna eventos e indices marcados."""
        serie = pd.to_numeric(self.df[nome], errors="coerce")
        serie = serie.replace([np.inf, -np.inf], np.nan)
        valores_validos = serie.dropna()

        # Sao necessarias pelo menos duas observacoes para calcular o desvio padrao.
        if len(valores_validos) < 2:
            return [], []

        media = valores_validos.mean()
        desvio = valores_validos.std(ddof=1)

        if pd.isna(desvio) or desvio == 0:
            return [], []

        limite = self.sensibilidade * desvio
        mascara = ((serie - media).abs() > limite).fillna(False)
        indices_anomalos = np.flatnonzero(mascara.to_numpy()).tolist()
        grupos = self.agrupar_indices(indices_anomalos)

        eventos = []
        for grupo in grupos:
            # Um representante por grupo: o ponto mais distante da media.
            indice_pico = max(
                grupo,
                key=lambda i: abs(serie.iloc[i] - media)
            )

            eventos.append({
                "indice": indice_pico,
                "referencia": self.df.iloc[indice_pico][self.coluna_referencia],
                "valor": float(serie.iloc[indice_pico]),
                "inicio": self.df.iloc[grupo[0]][self.coluna_referencia],
                "fim": self.df.iloc[grupo[-1]][self.coluna_referencia],
                "quantidade_pontos": len(grupo),
                "indices_grupo": grupo,
            })

        return eventos, indices_anomalos

    def detectar(self, parametros):
        """Executa a deteccao somente nos parametros escolhidos."""
        if isinstance(parametros, str):
            parametros = [parametros]

        if not parametros:
            raise ValueError("Selecione pelo menos um parametro para analisar.")

        ausentes = [nome for nome in parametros if nome not in self.df.columns]
        if ausentes:
            raise ValueError(f"Colunas nao encontradas no CSV: {ausentes}")

        self.resultados = {}
        self.pontos_anomalos = {}

        for nome in parametros:
            eventos, indices = self.detectar_parametro(nome)
            self.resultados[nome] = eventos
            self.pontos_anomalos[nome] = indices

        return self.resultados

    def criar_grafico(self, nome):
        """Cria um grafico do parametro pela coluna de referencia."""
        serie = pd.to_numeric(self.df[nome], errors="coerce")
        serie = serie.replace([np.inf, -np.inf], np.nan)
        eixo_x = self.df[self.coluna_referencia]

        figura = go.Figure()
        figura.add_trace(go.Scatter(
            x=eixo_x,
            y=serie,
            mode="lines",
            name=nome,
            connectgaps=False,
            line={"width": 1.7},
            hovertemplate=(
                f"{self.coluna_referencia}: %{{x}}<br>"
                f"{nome}: %{{y}}<extra></extra>"
            ),
        ))

        indices = self.pontos_anomalos.get(nome, [])
        if indices:
            figura.add_trace(go.Scatter(
                x=eixo_x.iloc[indices],
                y=serie.iloc[indices],
                mode="markers",
                name="Pontos acima do limite",
                marker={"color": "orange", "size": 8, "symbol": "circle"},
                hovertemplate=(
                    f"{self.coluna_referencia}: %{{x}}<br>"
                    f"{nome}: %{{y}}<br>Fora do limite estatistico"
                    "<extra></extra>"
                ),
            ))

        eventos = self.resultados.get(nome, [])
        if eventos:
            indices_picos = [evento["indice"] for evento in eventos]
            figura.add_trace(go.Scatter(
                x=eixo_x.iloc[indices_picos],
                y=serie.iloc[indices_picos],
                mode="markers",
                name="Representante de cada ocorrencia",
                marker={"color": "red", "size": 13, "symbol": "star"},
                customdata=[
                    [evento["inicio"], evento["fim"], evento["quantidade_pontos"]]
                    for evento in eventos
                ],
                hovertemplate=(
                    f"{self.coluna_referencia}: %{{x}}<br>"
                    f"{nome}: %{{y}}<br>"
                    "Inicio do grupo: %{customdata[0]}<br>"
                    "Fim do grupo: %{customdata[1]}<br>"
                    "Pontos anomalos no grupo: %{customdata[2]}"
                    "<extra></extra>"
                ),
            ))

        figura.update_layout(
            title=f"{nome} em funcao de {self.coluna_referencia}",
            xaxis_title=self.coluna_referencia,
            yaxis_title=nome,
            hovermode="closest",
            legend_title="Legenda",
            margin={"l": 20, "r": 20, "t": 60, "b": 20},
        )
        return figura

    def tabela_resultados(self):
        """Converte os eventos agrupados em uma tabela para exibicao/exportacao."""
        linhas = []
        for parametro, eventos in self.resultados.items():
            for evento in eventos:
                linhas.append({
                    "Parametro": parametro,
                    "Referencia da anomalia": evento["referencia"],
                    "Valor": evento["valor"],
                    "Inicio do grupo": evento["inicio"],
                    "Fim do grupo": evento["fim"],
                    "Quantidade de pontos anomalos": evento["quantidade_pontos"],
                })

        return pd.DataFrame(linhas)


def ler_csv(uploaded_file, separador, decimal):
    """Le o arquivo enviado, tentando UTF-8 e depois Latin-1."""
    conteudo = uploaded_file.getvalue()
    ultimo_erro = None

    for codificacao in ("utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(
                BytesIO(conteudo),
                sep=separador,
                decimal=decimal,
                encoding=codificacao,
                low_memory=False,
            )
        except UnicodeDecodeError as erro:
            ultimo_erro = erro

    raise ultimo_erro


def main():
    st.set_page_config(page_title="Detector de anomalias", layout="wide")
    st.title("Detector de anomalias em arquivos CSV")
    st.write(
        "Envie um CSV, escolha a coluna de referencia e selecione os "
        "parametros numericos que deseja analisar."
    )

    arquivo = st.file_uploader("Selecione o arquivo CSV", type=["csv", "txt"])
    if arquivo is None:
        st.info("Envie um arquivo para configurar a analise.")
        st.stop()

    with st.sidebar:
        st.header("Leitura do arquivo")
        opcoes_separador = {
            "Virgula ( , )": ",",
            "Ponto e virgula ( ; )": ";",
            "Tabulacao": "\t",
            "Barra vertical ( | )": "|",
        }
        nome_separador = st.selectbox("Separador de colunas", list(opcoes_separador))
        separador = opcoes_separador[nome_separador]
        decimal = st.selectbox("Separador decimal", [".", ","], index=0)

    try:
        df = ler_csv(arquivo, separador, decimal)
    except pd.errors.EmptyDataError:
        st.error("O arquivo enviado esta vazio.")
        st.stop()
    except pd.errors.ParserError as erro:
        st.error(f"Nao foi possivel interpretar o CSV: {erro}")
        st.stop()
    except (UnicodeDecodeError, ValueError) as erro:
        st.error(f"Erro ao ler o arquivo: {erro}")
        st.stop()

    df.columns = df.columns.astype(str).str.strip()
    if df.empty or len(df.columns) == 0:
        st.error("O arquivo nao contem linhas ou colunas para analisar.")
        st.stop()

    st.subheader("Pre-visualizacao dos dados")
    st.dataframe(df.head(10), use_container_width=True)
    st.caption(f"Arquivo: {arquivo.name} | Linhas: {len(df)} | Colunas: {len(df.columns)}")

    # Disponibiliza como parametros as colunas com pelo menos dois valores numericos.
    parametros_numericos = []
    for coluna in df.columns:
        valores = pd.to_numeric(df[coluna], errors="coerce")
        if np.isfinite(valores).sum() >= 2:
            parametros_numericos.append(coluna)

    if not parametros_numericos:
        st.error("Nao foram encontradas colunas com pelo menos dois valores numericos.")
        st.stop()

    with st.form("configuracao_analise"):
        st.subheader("Configuracao da analise")
        coluna_referencia = st.selectbox(
            "Parametro base (eixo horizontal dos graficos)",
            options=list(df.columns),
        )

        opcoes_parametros = [
            coluna for coluna in parametros_numericos
            if coluna != coluna_referencia
        ]
        parametros = st.multiselect(
            "Parametros que deseja analisar",
            options=opcoes_parametros,
            help="Selecione uma ou mais colunas numericas do arquivo.",
        )

        col1, col2 = st.columns(2)
        with col1:
            sensibilidade = st.number_input(
                "Sensibilidade (multiplicador do desvio padrao)",
                min_value=0.1,
                max_value=20.0,
                value=2.0,
                step=0.1,
            )
        with col2:
            max_gap = st.number_input(
                "max_gap (linhas normais permitidas entre pontos anomalos)",
                min_value=0,
                max_value=100000,
                value=1,
                step=1,
            )

        executar = st.form_submit_button("Analisar anomalias", type="primary")

    if executar:
        if not parametros:
            st.warning("Selecione pelo menos um parametro para analisar.")
            st.stop()

        try:
            analisador = AnalisadorAnomalias(
                dados=df,
                coluna_referencia=coluna_referencia,
                sensibilidade=sensibilidade,
                max_gap=max_gap,
            )
            analisador.detectar(parametros)
        except (ValueError, TypeError) as erro:
            st.error(str(erro))
            st.stop()

        tabela = analisador.tabela_resultados()
        quantidade_eventos = len(tabela)
        st.subheader("Resultados")
        st.metric("Ocorrencias agrupadas", quantidade_eventos)

        if tabela.empty:
            st.info("Nenhuma anomalia foi identificada com os criterios escolhidos.")
        else:
            st.dataframe(tabela, use_container_width=True)
            csv_saida = tabela.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "Baixar resultados em CSV",
                data=csv_saida,
                file_name="anomalias_detectadas.csv",
                mime="text/csv",
            )

        st.subheader("Graficos por parametro")
        st.caption(
            "Laranja: todos os pontos que ultrapassaram o limite estatistico. "
            "Vermelho: o ponto mais distante da media em cada grupo, usado "
            "como representante daquela ocorrencia."
        )

        for parametro in parametros:
            eventos = analisador.resultados[parametro]
            with st.expander(
                f"{parametro} — {len(eventos)} ocorrencia(s)",
                expanded=True,
            ):
                figura = analisador.criar_grafico(parametro)
                st.plotly_chart(figura, use_container_width=True)
                if eventos:
                    st.dataframe(
                        pd.DataFrame([
                            {
                                "Referencia": evento["referencia"],
                                "Valor": evento["valor"],
                                "Inicio": evento["inicio"],
                                "Fim": evento["fim"],
                                "Pontos anomalos": evento["quantidade_pontos"],
                            }
                            for evento in eventos
                        ]),
                        use_container_width=True,
                    )
                else:
                    st.write("Nenhuma ocorrencia encontrada para este parametro.")


if __name__ == "__main__":
    main()
