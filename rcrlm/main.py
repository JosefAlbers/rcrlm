from tokenizerz import Tokenizer
from .utils import load_model, load_config, download_repo, infer, train, collapse, distill, cascade, dampen, prune, drink, tie, deacon, fish
from .qwen3 import Qwen3ForCausalLM
from .tamo import TAMOQwen3, TamoConfig

ARCHS = dict(Qwen3ForCausalLM=Qwen3ForCausalLM,)

def test(task='all', num_repeat=1):
    if task == 'infer' or task == 'all':
        m = load()
        print('〄 Testing vanilla decoding...')
        _ = infer("Write a story about Einstein\n", **m, max_new_tokens=256)#, chat_template_kwargs=dict(enable_thinking=False))
        del m
    if task == 'batch' or task == 'all':
        m = load()
        print('〄 Testing batch decoding...')
        _ = infer(["#write a quick sort algorithm\n", "Give me a short introduction to large language model.\n", "Write a neurology ICU admission note.\n", "Comparison of Sortino Ratio for Bitcoin and Ethereum."], **m)
        del m
    if task == 'train' or task == 'all':
        m = load()
        lora_test_path = 'test_lora.safetensors'
        print('〄 Testing DoRA training...')
        train("RandomNameAnd6/SVGenerator", **m, lora_cfg=dict(wt_to=lora_test_path))
        del m
        print('〄 Testing DoRA decoding...')
        m = load()
        _ = infer("medium red circle", **m, lora_path=lora_test_path, stream=False, max_new_tokens=256, chat_template_kwargs=dict(enable_thinking=False))
        del m
    if task == 'collapse' or task == 'all':
        heal_test_path = 'test_heal.safetensors'
        for _test_collapse_idx in range(num_repeat):
            print(f'〄 Testing collapse {_test_collapse_idx}/{num_repeat}...')
            m = load()
            m['model'] = collapse(m['model'])#, do_rectify=True, do_quantize=True)
            print('✓ Colapsed:')
            _ = infer("Write a story about Einstein\n", **m, stream=False)
            teacher = load()['model']
            m['model'] = distill("HuggingFaceH4/instruction-dataset", **m, to=heal_test_path, teacher=teacher)
            print('✓ Healed:')
            _ = infer("Write a story about Einstein\n", **m, stream=False, max_new_tokens=1024)
            m['model'] = dampen(m['model'])
            print('✓ Dampened:')
            _ = infer("Write a story about Einstein\n", **m, stream=False, max_new_tokens=1024)
            del teacher, m
    if task == 'cascade' or task == 'all':
        heal_test_path = 'test_heal.safetensors'
        m = load()
        print('〄 Testing cascading...')
        teacher = load()['model']
        m['model'] = cascade("HuggingFaceH4/instruction-dataset", **m, to=heal_test_path, teacher=teacher)
        print('✓ Cascaded:')
        _ = infer("Write a story about Einstein\n", **m, stream=False, max_new_tokens=1024, limit_thinking=True)
        m['model'] = dampen(m['model'])
        print('✓ Dampened:')
        _ = infer("Write a story about Einstein\n", **m, stream=False, max_new_tokens=1024, limit_thinking=True)
    if task == 'drink' or task == 'all':
        print('〄 Testing drink...')
        m = load()
        m['model'] = drink(m['model'])
        _ = infer("Write a story about Einstein\n", **m, stream=False)
        teacher = load()['model']
        m['model'] = distill("HuggingFaceH4/instruction-dataset", **m, teacher=teacher, unfreeze_all=True)
        _ = infer("Write a story about Einstein\n", **m, stream=False)
        del m
    if task == 'prune' or task == 'all':
        print('〄 Testing pruning...')
        m = load()
        m['model'] = prune("HuggingFaceH4/instruction-dataset", **m)
        _ = infer("Write a story about Einstein\n", **m, stream=False, max_new_tokens=1024, limit_thinking=True)
        del m
    if task == 'deacon' or task == 'all':
        m = load()
        print('〄 Testing deacon...')
        m['model'] = deacon(**m)
        _ = infer("Write a story about Einstein\n", **m, max_new_tokens=256)#, chat_template_kwargs=dict(enable_thinking=False))
        teacher = load()['model']
        m['model'] = distill("HuggingFaceH4/instruction-dataset", **m, teacher=teacher, add_dora=False)
        print('✓ Healed:')
        _ = infer("Write a story about Einstein\n", **m, stream=False, max_new_tokens=1024)
        del m, teacher
    if task == 'tie' or task == 'all':
        m = load()
        print('〄 Testing tie...')
        m['model'] = tie(**m)
        _ = infer("Write a story about Einstein\n", **m, max_new_tokens=256)#, chat_template_kwargs=dict(enable_thinking=False))
        del m
    if task == 'tamo':# or task == 'all':
        print('〄 Testing tamo...')
        m = load()
        model_tamo = TAMOQwen3(m['config'])
        model_tamo.llm = m['model']
        del m, model_tamo
    if task == 'fish':# or task == 'all':
        m = load()
        print('〄 Testing fish...')
        m['model'] = fish(**m, calib_ds_id="HuggingFaceH4/instruction-dataset")
        _ = infer("Write a story about Einstein\n", **m, max_new_tokens=256)#, chat_template_kwargs=dict(enable_thinking=False))
        del m
    if task == 'eval' or task == 'all':
        eval_limit=20
        try:
            print('〄 Testing lm-eval...')
            print('{{{')
            heal_test_path = 'test_heal.safetensors'
            eval_str = ''
            from .evals import eval_lm
            m = load()
            eval_str += f'✓ Original:\n{eval_lm(**m, limit=eval_limit)}\n'
            m['model'] = collapse(m['model'])
            eval_str += f'✓ Collapsed:\n{eval_lm(**m, limit=eval_limit)}\n'
            teacher = load()['model']
            m['model'] = distill("HuggingFaceH4/instruction-dataset", **m, teacher=teacher)
            eval_str += f'✓ Healed:\n{eval_lm(**m, limit=eval_limit)}\n'
            m['model'] = dampen(m['model'])
            eval_str += f'✓ Dampened:\n{eval_lm(**m, limit=eval_limit)}\n'
            print('}}}')
            print(eval_str)
            del teacher, m
        except Exception as e:
            print(e)
            print('[ERROR] Need to pip install lm-eval first')

def load(model_id='Qwen/Qwen3-0.6B', extra_config=None):
# def load(model_id='Qwen/Qwen3-4B-Instruct-2507', extra_config=None):
    repo_name, model_name = model_id.split('/')
    model_path = download_repo(repo_name, model_name)
    model_cfg = load_config(model_path)
    if extra_config and isinstance(extra_config, dict):
        model_cfg.extra_config = extra_config
    model_cls = ARCHS.get(model_cfg.architectures[0])
    model = load_model(model_cls, model_path, model_cfg)
    tokenizer = Tokenizer(repo_name='local', model_name=model_path)
    return dict(model=model, tokenizer=tokenizer, config=model_cfg)

def cli():
    import argparse
    parser = argparse.ArgumentParser(description="Load a model and generate text.")
    parser.add_argument("-m", "--model", type=str, default='Qwen/Qwen3-0.6B', dest="model_id", help="Model ID in the format 'repo/model_name'.")
    parser.add_argument("-p", "--prompts", type=str, nargs='*', help="Prompt(s) for generation.")
    parser.add_argument("-n", "--new", type=int, default=100, help="Maximum new tokens to generate.")
    parser.add_argument("-s", "--scan", action="store_true", help="Enable scan mode.")
    parser.add_argument("-j", "--jit", action="store_true", help="Enable JIT compilation.")
    parser.add_argument("--no-format", dest="use_chat_template", action="store_false", help="Do not use chat template.")
    parser.add_argument("--no-stream", dest="stream", action="store_false", help="Do not stream output.")
    args =  parser.parse_args()
    if args.prompts:
        args.prompts = [p.replace("\\n", "\n") for p in args.prompts]
    else:
        args.prompts = "Give me a short introduction to large language model.\n"
    m = load(args.model_id)
    _ = infer(
        prompts=args.prompts,
        **m,
        max_new_tokens=args.new,
        use_chat_template=args.use_chat_template,
        stream=args.stream,
        use_scan=args.scan,
        use_jit=args.jit
    )

if __name__ == "__main__":
    test()
