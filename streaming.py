import time, torch, gc, asyncio
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams

model_path = "/srv/models/qwen2.5-0.5B"

prompt = "What was the main work done by Da Vinci?"

engine_args = AsyncEngineArgs(
    model=model_path,
    dtype="float16",
    tensor_parallel_size=1,
    gpu_memory_utilization=0.9,
    max_num_batched_tokens=32768,
    max_num_seqs=256,
    disable_log_stats=True,
)

# create vLLM async streaming engine
engine = AsyncLLMEngine.from_engine_args(engine_args)

# async func to generate text in streaming mode
async def generate_text(prompt: str, max_tokens: int = 100):
    try:
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=max_tokens,
            stop = ['\n']
        )
        request_id = "test-request"
        results_generator = engine.generate(
            prompt=prompt,
            sampling_params=sampling_params,
            request_id=request_id,
        )
        final_output = None
        async for request_output in results_generator:
            final_output = request_output
            for output in request_output.outputs:
                print(output.text, end="", flush=True)
            print()
        return final_output
    except asyncio.CancelledError:
        print("\nGeneration was cancelled")
        return None
    finally:
        try:
            await engine.abort(request_id)
        except:
            pass

res = asyncio.run(generate_text(prompt))
print(res.outputs[0].text)
