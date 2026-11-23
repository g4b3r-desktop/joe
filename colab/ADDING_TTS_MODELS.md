# Adicionar novos modelos e idiomas ao Colab

Este guia explica como ampliar o suporte de modelos TTS no notebook do JOE. Ele é voltado a quem altera o código do projeto; **não basta copiar um checkpoint para o Google Drive**. O notebook, a sincronização do ZIP e a API precisam reconhecer a nova configuração.

> **Estado atual desta branch:** a sincronização automática de `colab/drive_model_sync.py` está implementada especificamente para as variantes `portuguese` e `portuguese_24l`. A função `ensure_portuguese_model_in_zip()` não descobre idiomas automaticamente. Inglês, japonês ou outro idioma só estarão disponíveis no modo Colab depois de integrar seus arquivos e configurações nas etapas deste documento.

## 1. Identifique o modelo antes de alterar o código

Para o idioma/modelo que deseja adicionar, registre:

- Nome estável usado pelo JOE, por exemplo `english` ou `japanese`.
- Repositório oficial/comunitário no Hugging Face.
- Arquivos exatos necessários: pesos, tokenizer, configuração YAML e, se aplicável, pesos sem clonagem e áudio de referência.
- Revisão imutável (commit SHA) de cada repositório.
- Variante de arquitetura: número de camadas, formato dos pesos e compatibilidade com a versão instalada do Pocket TTS.
- Licença, atribuição obrigatória e eventuais condições de acesso.
- Tamanho dos arquivos e espaço final estimado do ZIP.

Não presuma que todo idioma tem uma variante `24l`, que usa o mesmo tokenizer ou que pode compartilhar os pesos de outro idioma. Modelos comunitários também podem ter estrutura, dependências e licenças diferentes.

## 2. Defina a estrutura dentro do ZIP

Mantenha uma estrutura previsível e sem misturar arquivos de idiomas diferentes. Exemplo ilustrativo:

```text
JOE_MODELS/
├── pocket-tts/
│   └── languages/
│       ├── portuguese/
│       │   └── model.safetensors
│       └── english/
│           └── model.safetensors
└── pocket-tts-without-voice-cloning/
    └── languages/
        ├── portuguese/
        │   └── tokenizer.json
        └── english/
            └── tokenizer.json
```

A estrutura acima é um exemplo, não uma afirmação de que todos os repositórios publicam exatamente esses arquivos. Confirme os caminhos no YAML e no repositório de origem antes de cadastrar os assets.

Para modelos com clonagem e sem clonagem, inclua cada arquivo somente se o modelo realmente o fornecer e se a implementação precisar dele. Não crie caminhos fictícios para completar a estrutura.

## 3. Cadastre os arquivos na sincronização do ZIP

Arquivo: `colab/drive_model_sync.py`

Atualmente, `MODEL_ASSETS` relaciona o caminho interno do ZIP a uma tupla:

```python
"caminho/dentro/do/zip": (
    "proprietario/repositorio-huggingface",
    "caminho/do/arquivo/no-repositorio",
    "commit-sha-imutavel",
    "safetensors",  # ou "json", conforme o validador disponível
)
```

Adicione uma entrada para cada arquivo obrigatório do novo modelo. Fixe a revisão pelo commit SHA; evite usar `main` ou `latest`, pois o conteúdo poderia mudar entre sessões.

**Importante:** apenas adicionar entradas a `MODEL_ASSETS` não habilita outro idioma nesta branch. A função `ensure_portuguese_model_in_zip()` ainda tem nome, documentação e comportamento voltados às duas variantes portuguesas. Ao ampliar o suporte:

1. Generalize a função para receber a lista de idiomas/variantes solicitada, ou crie uma função genérica equivalente.
2. Faça a seleção dos assets a sincronizar depender dessa lista; não baixe automaticamente todos os idiomas.
3. Preserve os membros desconhecidos do ZIP, como a rotina atual faz.
4. Mantenha a validação de cada arquivo baixado e de cada membro final do ZIP.
5. Continue usando arquivo temporário e substituição atômica: o ZIP original deve permanecer intacto se download, validação ou gravação falhar.
6. Respeite `MAX_ARCHIVE_BYTES` (4.000.000.000 bytes). Se a atualização ultrapassar 4 GB, interrompa sem substituir o ZIP anterior.
7. Retorne claramente quais arquivos foram reutilizados e quais foram baixados.

Para Safetensors, use a validação de cabeçalho/estrutura existente. Para JSON, confirme que o conteúdo é JSON válido e tem a estrutura esperada para o tokenizer. Não classifique um arquivo apenas pela extensão.

## 4. Prepare e valide o YAML do Pocket TTS

O notebook usa os YAMLs fornecidos pela instalação do pacote Pocket TTS como base e grava cópias locais em `/content/joe/runtime/`.

Para cada novo idioma:

1. Localize o YAML correspondente na pasta `config` do pacote instalado.
2. Confira os nomes e a estrutura das chaves. Não copie cegamente o YAML de português se o modelo tiver configuração diferente.
3. Aponte `weights_path` para o checkpoint extraído do ZIP.
4. Aponte `flow_lm.lookup_table.tokenizer_path` para o tokenizer correto, se esse campo fizer parte da configuração do modelo.
5. Configure `weights_path_without_voice_cloning` somente quando houver um checkpoint compatível.
6. Preserve parâmetros específicos do modelo, como arquitetura, número de camadas, dimensões e opções do tokenizer.

A função `validate_local_tts_configs()` já recebe um dicionário de configurações e valida os caminhos de pesos e tokenizer. Inclua a nova configuração nesse dicionário. Se o novo modelo exigir outro formato de arquivo ou validações específicas, implemente um validador próprio em vez de ignorar a validação.

## 5. Atualize a célula de configuração do notebook

Arquivo: `colab/Joe_Colab_API.ipynb`, célula **Montar o Google Drive e configurar os modelos de voz**.

A célula atual tem caminhos explícitos para `portuguese` e `portuguese_24l`. Ao adicionar um idioma:

- Inclua o idioma na seleção de modelos que o usuário deseja instalar.
- Passe essa seleção para a rotina genérica de sincronização.
- Extraia e localize os arquivos do idioma selecionado.
- Gere um YAML local para cada idioma/variante selecionado.
- Valide todas as configurações antes de habilitar `JOE_TTS_USE_DRIVE_MODELS=1).
- Mostre no output os arquivos encontrados, baixados, reutilizados e validados.
- Se qualquer modelo selecionado falhar, limpe as flags de configuração e impeça a inicialização da API em modo Drive.
- Não apague uma extração válida sem necessidade. Se o ZIP mudar, invalide somente o que não puder mais ser reutilizado com segurança.

Não mantenha variáveis com nomes exclusivos de português como solução definitiva. Prefira um registro/dicionário de idiomas e variantes, por exemplo `configs["english"]` e `configs["japanese"]`.

## 6. Faça a API reconhecer o novo idioma

A configuração do notebook não é suficiente se a API rejeitar o idioma ou continuar escolhendo somente os YAMLs portugueses.

Localize no código da API Colab:

- a lista de idiomas aceitos;
- o resolvedor que escolhe a configuração TTS;
- o carregamento/cache do modelo;
- a validação do campo `language` recebido em `POST /v1/jobs/tts`;
- a seleção de variante, caso o modelo tenha mais de uma.

Adicione o novo identificador e faça o resolvedor buscar a configuração validada correspondente. Evite caminhos fixos espalhados pelo código. O idioma recebido na requisição deve corresponder a uma configuração que foi efetivamente preparada no runtime; caso contrário, retorne um erro claro, sem fallback silencioso para português.

Se o JOE desktop ainda não oferecer esse idioma no seletor, será necessário adicioná-lo também à lista de idiomas do cliente. A API Colab não consegue usar um idioma que o cliente não permita selecionar/enviar.

## 7. Credenciais, licença e tamanho

- Se o repositório exigir autenticação, use o Secret `HF_TOKEN` do Colab com **Notebook access** habilitado.
- Nunca grave o token no notebook, no ZIP ou em arquivos versionados.
- Registre no documento do modelo a licença e a atribuição exigida.
- Verifique se o uso e a redistribuição dos pesos no ZIP são permitidos pela licença.
- Considere o limite de 4 GB do ZIP e o espaço disponível no Google Drive e no runtime.
- Pesos Safetensors normalmente já são compactados; não conte com grande redução ao comprimi-los novamente.

## 8. Testes obrigatórios

Antes de considerar o idioma pronto, adicione testes para:

1. ZIP sem o novo idioma: baixar os arquivos necessários.
2. ZIP com arquivos válidos: reutilizar sem novo download.
3. Arquivo ausente, duplicado, truncado ou inválido: substituir somente o necessário.
4. Erro de download ou token sem permissão: manter o ZIP original intacto.
5. Limite de 4 GB excedido: falhar sem substituir o ZIP original.
6. YAML com pesos ou tokenizer ausente: bloquear a inicialização.
7. Seleção de dois ou mais idiomas: preparar todos os selecionados sem misturar arquivos.
8. API: aceitar o novo idioma quando configurado e rejeitá-lo com mensagem clara quando não estiver instalado.
9. Requisição TTS real no runtime Colab, usando o idioma e a variante adicionados.

Execute pelo menos:

```bash
python -m pytest tests/test_colab_drive_model_sync.py -q
python -m pytest -q
```

Depois, execute o notebook do início ao fim em um runtime Colab novo. Um teste local da sincronização não substitui a validação de inferência no Colab.

## Checklist para cada novo modelo

- [ ] Repositório, commit SHA, licença e atribuição documentados.
- [ ] Arquivos necessários identificados no YAML de origem.
- [ ] Assets adicionados à sincronização genérica.
- [ ] Download seletivo implementado; não baixa idiomas não selecionados.
- [ ] Limite de 4 GB e preservação do ZIP original testados.
- [ ] YAML local criado com os caminhos corretos.
- [ ] Checkpoint e tokenizer validados.
- [ ] Célula do notebook atualizada.
- [ ] API reconhece o idioma e sua variante.
- [ ] Seletor do JOE desktop atualizado, quando necessário.
- [ ] Testes automatizados e inferência real no Colab aprovados.

## Arquivos principais

| Arquivo | Responsabilidade |
|---|---|
| `colab/drive_model_sync.py` | Download, validação e atualização segura do ZIP do Drive |
| `colab/safetensors_validation.py` | Diagnóstico/validação de arquivos Safetensors |
| `colab/Joe_Colab_API.ipynb` | Montagem do Drive, seleção, extração, YAMLs e inicialização da API |
| `tests/test_colab_drive_model_sync.py` | Testes da sincronização dos modelos |

Este guia descreve o trabalho necessário para integrar novos modelos. Ele não significa que inglês ou japonês já estejam habilitados na sincronização/API desta branch.
