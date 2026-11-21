import time, sys
from llmlingua import PromptCompressor
t=time.time(); c = PromptCompressor(model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank", use_llmlingua2=True, device_map="cpu"); print("load", round(time.time()-t,1), "s")
txt = open(sys.argv[1], encoding="utf-8").read()
import re
prose = re.sub(r"```.*?```", "", txt, flags=re.S)[:60000]
t=time.time(); r = c.compress_prompt(prose, rate=0.33, force_tokens=["\n","."], drop_consecutive=True); dt=time.time()-t
print("words", len(prose.split()), "->", len(r["compressed_prompt"].split()), "tokens", r["origin_tokens"], "->", r["compressed_tokens"], f"in {dt:.1f}s = {r['origin_tokens']/dt:.0f} tok/s")
print("SAMPLE:", r["compressed_prompt"][2000:2400])
