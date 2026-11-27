# Backends avançados e otimizações experimentais do Pocket TTS

Este documento descreve experimentos isolados para acelerar o Pocket TTS no Joe. **Nenhuma dessas opções muda o backend padrão do aplicativo.** O caminho oficial do Joe continua sendo Pocket TTS/PyTorch, com seleção CPU/CUDA e quantização INT8 somente em CPU.

## Estado atual no Joe

O desktop mantém uma instância do modelo e estados de voz em cache. A inferência local seleciona CPU ou CUDA; o INT8 dinâmico é usado apenas no caminho CPU. O Colab tem seu próprio carregador e worker sequencial. Portanto, qualquer backend alternativo precisa funcionar tanto com os modelos/variantes portugueses usados pelo Joe quanto com o fluxo de voz de referência, ou ser claramente limitado a um modo de teste separado.

A prioridade é manter estas propriedades:

- Mesmo idioma, variante, texto e gravação de referência ao comparar runtimes.
- Preparar novamente o estado de voz para cada repetição de benchmark.
- Separar tempo de carregamento, preparação da voz, primeira chamada (incluindo compilação JIT) e geração aquecida.
- Sincronizar CUDA antes de parar os cronômetros.
- Validar áudio finito, não vazio, duração razoável e qualidade perceptiva.
- Só promover uma otimização ao app depois de comparar vários textos curtos/longos e gravações de voz reais.

## Experimento incluído: `torch.compile` e precisão mista

O script `scripts/benchmark_pocket_tts_experimental.py` compara o baseline com uma execução experimental separada. Ele não modifica o serviço `PocketTTSService` nem as configurações persistentes do Joe.

### 1. Inspecionar módulos compiláveis

Ative o ambiente do Joe e execute na raiz do repositório:

```bash
source .venv/bin/activate
python scripts/benchmark_pocket_tts_experimental.py \
  --device cuda --language portuguese --variant portuguese --list-modules
```

Escolha um caminho de submódulo listado. Não presuma que um nome de módulo será igual entre versões do Pocket TTS.

### 2. Testar compilação seletiva

```bash
python scripts/benchmark_pocket_tts_experimental.py \
  --device cuda \
  --language portuguese --variant portuguese \
  --compile-module CAMINHO_LISTADO \
  --compile-mode default \
  --repeats 5 --warmup 1 \
  --output /tmp/joe-pocket-tts-compile.wav \
  --json-out /tmp/joe-pocket-tts-compile.json
```

O script reporta a primeira chamada separadamente da média aquecida. A compilação JIT pode custar segundos ou minutos e pode recompilar se encontrar formas ou guardas diferentes. O modo dinâmico automático padrão do PyTorch deve ser testado antes de forçar dimensões dinâmicas; `dynamic=True` global não é uma recomendação geral. Para investigar recompilações, tente:

```bash
TORCH_LOGS=recompiles,dynamic python scripts/benchmark_pocket_tts_experimental.py \
  --device cuda --compile-module CAMINHO_LISTADO
```

Comece por um único bloco com bastante computação e poucas mudanças de forma. Não compile todo o método de geração autoregressiva às cegas. Se houver erros de graph break, recompilações contínuas ou queda de velocidade depois do aquecimento, descarte o experimento.

### 3. Testar FP16, BF16 e TF32 separadamente

FP16:

```bash
python scripts/benchmark_pocket_tts_experimental.py \
  --device cuda --autocast fp16 --repeats 5 --warmup 1 \
  --output /tmp/joe-pocket-tts-fp16.wav
```

BF16 (somente quando a GPU/build informar suporte):

```bash
python scripts/benchmark_pocket_tts_experimental.py \
  --device cuda --autocast bf16 --repeats 5 --warmup 1
```

TF32:

```bash
python scripts/benchmark_pocket_tts_experimental.py \
  --device cuda --tf32 --repeats 5 --warmup 1
```

Não combine FP16, BF16, TF32 e compilação no primeiro ensaio. Primeiro descubra qual mudança, isoladamente, ajuda. Depois teste combinações. O relatório estima diferença numérica entre as amostras dos dois WAVs, mas isso **não** garante identidade auditiva: escute os arquivos e confira pronúncia, ruído, cortes, silêncio e naturalidade. O script usa a mesma seed para reduzir a variabilidade do sampler, mas a igualdade numérica depende da implementação.

## Alternativas em avaliação

### A. Pocket TTS Rust/Candle

O port comunitário originalmente anunciado em [kyutai-labs/pocket-tts issue #43](https://github.com/kyutai-labs/pocket-tts/issues/43) está hoje no repositório [stepupgaming/pocket-tts](https://github.com/stepupgaming/pocket-tts). O README atual descreve uma implementação Rust/Candle, CLI, serviço HTTP, entrada de voz WAV e bindings Python via Maturin.

Os números de aproximadamente 3,57× (texto curto) e 2,44× (texto longo) foram relatados pelo autor do port na issue, em testes próprios; não são benchmark oficial da Kyutai nem uma expectativa para a RTX 3060 ou para o hardware do usuário. O README do port também apresenta resultados posteriores medidos em Apple M4 Max, que não devem ser comparados diretamente com o Joe sem reproduzir o mesmo teste.

**Bloqueio antes de integrar ao Joe:** não foi confirmada compatibilidade do port com as variantes portuguesas do Joe (`portuguese` e `portuguese_24l`), com os YAMLs/checkpoints/tokenizers enviados ao Drive, nem com estados de voz `.safetensors` já preparados pelo Joe. A API de exemplo do port aceita uma gravação WAV como referência, o que não prova compatibilidade binária com os estados de voz do backend Python. Também é preciso testar acentos, pontuação, textos longos e vozes personalizadas em português brasileiro.

Plano de validação:
1. Fixar um commit do port e registrar versão do Rust/Candle e modelo carregado.
2. Carregar o mesmo checkpoint/tokenizer português (sem converter silenciosamente os pesos).
3. Testar clonagem com uma gravação WAV e comparar com o Python.
4. Testar se é possível exportar/importar um estado de voz sem perda ou, inicialmente, preparar a voz novamente a partir do WAV.
5. Rodar o corpus de português curto/médio/longo e medir TTFA, RTF, RAM e qualidade.
6. Só então decidir entre subprocesso Rust, binding Python ou serviço local.

Não adicione o pacote Rust como dependência obrigatória do Joe nesta etapa. Mantenha-o como backend opcional até completar a matriz de compatibilidade.

### B. `torch.compile`

Pode acelerar submódulos repetitivos, mas o Pocket TTS combina geração sequencial, estados internos e formatos que podem variar com o texto. O custo de compilação inicial deve ser contabilizado separadamente e o ganho precisa existir depois do aquecimento. O script deste branch permite selecionar um módulo específico em vez de aplicar o compilador indiscriminadamente ao pipeline inteiro.

### C. ONNX Runtime / OpenVINO

Não há uma conversão completa pronta para o pipeline atual do Joe. A geração autoregressiva, o estado de voz/cache e o decodificador de áudio precisam ser exportados e preservados corretamente. Uma investigação futura deve começar por um submódulo estável, com formas e entradas/saídas documentadas, e comparar numericamente com o PyTorch. Não substituir o serviço nem adicionar runtime ao instalador antes de um protótipo funcional.

### D. CUDA Graphs

Não capturar `generate_audio()` inteiro como primeira tentativa. Captura e replay de CUDA Graphs favorecem fluxos com buffers, formas e sequência de operações estáveis; o loop autoregressivo e a duração variável do áudio tornam o pipeline completo um alvo arriscado. A pesquisa deve isolar um passo tensorial repetido, pré-alocar buffers e validar que o estado não é compartilhado incorretamente entre gerações.

### E. Precisão mista

FP16/BF16 e TF32 devem ser avaliados de forma separada. A disponibilidade do tipo numérico não garante que todas as operações do modelo mantenham qualidade ou estabilidade. O teste deve rejeitar NaN/Inf, saída silenciosa e áudio truncado; depois comparar pronúncia e naturalidade ouvindo os WAVs. Não torne precisão reduzida padrão apenas com base em uso de GPU.

## Critérios para promover uma otimização

Uma opção só pode sair do modo experimental se:

1. O baseline e o candidato usarem o mesmo checkpoint, tokenizer, voz, texto, sampler e número de passos.
2. Houver ao menos 5 repetições aquecidas para texto curto, médio e longo.
3. A medição separar carregamento, preparação da voz, primeira chamada e geração.
4. O WAV for válido, sem NaN/Inf, clipping novo relevante ou truncamento.
5. A qualidade for ouvida por uma pessoa e considerada aceitável em português brasileiro.
6. O ganho for reproduzível e relevante no hardware de destino.
7. O fallback para o backend PyTorch atual continuar disponível.

## Referências

- [Pocket TTS da Kyutai](https://github.com/kyutai-labs/pocket-tts)
- [Issue do port Rust/Candle e números iniciais do autor](https://github.com/kyutai-labs/pocket-tts/issues/43)
- [Port Rust/Candle e bindings Python](https://github.com/stepupgaming/pocket-tts)
- [torch.compile — documentação PyTorch](https://docs.pytorch.org/docs/stable/generated/torch.compile.html)
- [Dynamic Shapes — documentação PyTorch](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_dynamic_shapes.html)
- [AMP / autocast — documentação PyTorch](https://docs.pytorch.org/docs/stable/amp.html)
- [CUDA Graphs — documentação PyTorch](https://docs.pytorch.org/docs/stable/notes/cuda.html#cuda-graphs)
