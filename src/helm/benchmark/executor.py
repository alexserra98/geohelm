from typing import Optional
from dataclasses import dataclass, replace

from helm.common.general import parallel_map
from helm.common.hierarchical_logger import htrack, hlog
from helm.common.request import RequestResult, Sequence
from helm.common.authentication import Authentication
from helm.proxy.services.remote_service import RemoteService
from helm.proxy.services.server_service import ServerService
from helm.proxy.services.service import Service
from helm.benchmark.adaptation.scenario_state import ScenarioState
from helm.benchmark.adaptation.request_state import RequestState
from functools import partial
from copy import deepcopy
import torch
from dataclasses import asdict
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from typing import Any, Dict, List

from helm.common.cache import Cache, CacheConfig
from helm.common.hierarchical_logger import htrack_block, hlog
from helm.common.request import EMBEDDING_UNAVAILABLE_REQUEST_RESULT, Request, RequestResult, Sequence, Token
from helm.common.tokenization_request import (
    TokenizationRequest,
    TokenizationRequestResult,
    DecodeRequest,
    DecodeRequestResult,
    TokenizationToken,
)
from helm.proxy.clients.client import Client, wrap_request_time, truncate_sequence, cleanup_tokens
from helm.proxy.clients.huggingface_tokenizer import HuggingFaceTokenizers
from helm.proxy.clients.huggingface_model_registry import (
    get_huggingface_model_config,
    HuggingFaceModelConfig,
    HuggingFaceHubModelConfig,
    HuggingFaceLocalModelConfig,
)
from threading import Lock

from helm.benchmark.hidden_geometry.utils import hidden_states_process

from einops import reduce
import numpy as np
import copy
from tqdm import tqdm
import pickle


class ExecutorError(Exception):
    pass


@dataclass(frozen=True)
class ExecutionSpec:
    # If non-empty, URL of the proxy server we send requests to (e.g., http://localhost:1959).
    url: Optional[str]

    # Pass into the service
    auth: Authentication

    # Path where API credentials and cache is stored.
    # This path is the same as `--base-path` when launching the proxy server (see server.py).
    # Required when url is not set.
    local_path: Optional[str]

    # How many threads to have at once
    parallelism: int

    # Whether to skip execution
    dry_run: bool = False

    # URL to the MongoDB database.
    # If non-empty, the MongoDB database will be used for caching instead of SQLite.
    # Example format: mongodb://[username:password@]host1[:port1]/[dbname]
    # For full format, see: https://www.mongodb.com/docs/manual/reference/connection-string/
    mongo_uri: str = ""


class Executor:
    """
    An `Executor` takes a `ScenarioState` which has a bunch of requests.
    Issue them to the API and return the results.
    """

    def __init__(self, execution_spec: ExecutionSpec):
        self.execution_spec = execution_spec

        self.service: Service
        if execution_spec.url:
            hlog(f"Running using remote API proxy server: {execution_spec.url}")
            self.service = RemoteService(execution_spec.url)
        elif execution_spec.local_path:
            hlog(f"Running in local mode with base path: {execution_spec.local_path}")
            self.service = ServerService(
                base_path=execution_spec.local_path, root_mode=True, mongo_uri=execution_spec.mongo_uri, 
            )
        else:
            raise ValueError("Either the proxy server URL or the local path must be set")

    @htrack(None)
    def execute(self, scenario_state: ScenarioState) -> ScenarioState:
        if self.execution_spec.dry_run:
            hlog("Skipped execution.")
            return scenario_state

        # Do it!
        process = partial(self.process, hidden_states=scenario_state.adapter_spec.hidden_states)
        # request_states = parallel_map(
        #     process,
        #     scenario_state.request_states,
        #     parallelism=self.execution_spec.parallelism,
        # )
        #using tqdm
        # model_kwargs = {}
        # model_kwargs["output_hidden_states"] = True
        # model_kwargs["device_map"]="auto"
        # model_kwargs["cache_dir"]="/orfeo/scratch/dssc/zenocosini/"
        # model_name = "meta-llama/Llama-2-13b-hf"
        # model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, **model_kwargs)
        # tokenizer = AutoTokenizer.from_pretrained(model_name, **model_kwargs)
        # request_states = [process(model, tokenizer, device,request_state) for request_state in tqdm(scenario_state.request_states, desc="Processing requests")]
        request_states = [request_state for request_state in tqdm(scenario_state.request_states, desc="Processing requests")]
        prompts = [request_state.request.prompt for request_state in request_states]
        raw_request = {
            "temperature": 1e-7 if request_states[0].request.temperature == 0 else request_states[0].request.temperature,
            "num_return_sequences": request_states[0].request.num_completions,
            "max_new_tokens": request_states[0].request.max_tokens,
            "top_p": request_states[0].request.top_p,
            "output_hidden_states": True,
        }
        
        with open("relevant_raw_request.pkl","wb") as f:
            pickle.dump(raw_request,f)
        with open("prompts.pkl", "wb") as f:
            pickle.dump(prompts, f)
        print(f'Request saved')
        for i,request_state in tqdm(enumerate(scenario_state.request_states), desc="Processing requests"):
            out = process(request_state)
            if out != -1:
                request_states.append(out)
            else:
                del scenario_state.request_states[i]    
        
        hlog(f"Processed {len(request_states)} requests")
        return ScenarioState(scenario_state.adapter_spec, request_states)


    # horrible to add hidden_geoemtry here but otherwise it require a major refactoring
    def process(self, state: RequestState, hidden_states = False) -> RequestState:
        try:
            result: RequestResult = self.service.make_request(self.execution_spec.auth, state.request, hidden_states=hidden_states)

        except Exception as e:
            raise ExecutorError(f"{str(e)} Request: {state.request}") from e
        if not result.success:
            if result.error_flags and not result.error_flags.is_fatal:
                hlog(f"WARNING: Non-fatal error treated as empty completion: {result.error}")
                result.completions = [Sequence(text="", logprob=0, tokens=[])]
            else:
                return -1
                #raise ExecutorError(f"{str(result.error)} Request: {state.request}")
        return replace(state, result=result)
    
    
#     def process(self, model,tokenizer,device, state: RequestState, hidden_states = False ) -> RequestState:
#             try:
                
#                 result: RequestResult = make_request(model,tokenizer,device,state.request, hidden_states=hidden_states)

#             except Exception as e:
#                 raise ExecutorError(f"{str(e)} Request: {state.request}") from e
#             if not result.success:
#                 if result.error_flags and not result.error_flags.is_fatal:
#                     hlog(f"WARNING: Non-fatal error treated as empty completion: {result.error}")
#                     result.completions = [Sequence(text="", logprob=0, tokens=[])]
#                 else:
#                     raise ExecutorError(f"{str(result.error)} Request: {state.request}")
#             return replace(state, result=result)
# def make_request(model,tokenizer,device,request: Request, **kwargs) -> RequestResult:

#         # Embedding not supported for this model
#         if request.embedding:
#             return EMBEDDING_UNAVAILABLE_REQUEST_RESULT

#         # Only a single stop sequence is supported as we can only pass in a single value for `eos_token_id`
#         if len(request.stop_sequences) > 1:
#             raise ValueError("More than one stop sequence is not supported.")

#         raw_request = {
#             "engine": request.model_engine,
#             "prompt": request.prompt,
#             "temperature": 1e-7 if request.temperature == 0 else request.temperature,
#             "num_return_sequences": request.num_completions,
#             "max_new_tokens": request.max_tokens,
#             "top_p": request.top_p,
#             "echo_prompt": request.echo_prompt,
#             "top_k_per_token": request.top_k_per_token,
#             "stop_sequences": request.stop_sequences,
#             "output_hidden_states": kwargs.get("hidden_states", False)
#         }

        


#         response = serve_request(raw_request, model, tokenizer, device)
#         completions = []
#         for raw_completion in response["completions"]:
#             sequence_logprob: float = 0
#             tokens: List[Token] = []
#             if request.echo_prompt:
#                 # Add prompt to list of generated tokens.
#                 generated_tokens = raw_completion["tokens"][response["input_length"] :]
#                 for token_text in raw_completion["tokens"][: response["input_length"]]:
#                     tokens.append(Token(text=token_text, logprob=0.0, top_logprobs={}))
#             else:
#                 generated_tokens = raw_completion["tokens"]

#             # Compute logprob for the entire sequence.
#             for token_text, logprob, top_logprobs_dict in zip(
#                 generated_tokens, raw_completion["logprobs"], raw_completion["top_logprobs_dicts"]
#             ):
#                 tokens.append(Token(text=token_text, logprob=logprob, top_logprobs=top_logprobs_dict))
#                 sequence_logprob += logprob
            
#             #modifying the hidden states so to keep only the last token and the avg of the tokens of the question
#             hidden_states = raw_completion["hidden_states"]

#             completion = Sequence(text=raw_completion["text"], logprob=sequence_logprob, tokens=tokens, hidden_states=hidden_states)
#             completion = truncate_sequence(completion, request)
#             completions.append(completion)

#         return RequestResult(
#             success=True,
#             cached=False,
#             request_time=0.0,
#             request_datetime=response.get("request_datetime"),
#             completions=completions,
#             embedding=[],
#         )
        
# def serve_request(raw_request: Dict[str, Any], model, tokenizer, device):
#     #-----------------------------------------
#     encoded_input = tokenizer(raw_request["prompt"], return_tensors="pt", return_token_type_ids=False).to(
#         device
#     )

#     raw_request = deepcopy(raw_request)
#     raw_request["do_sample"] = True
#     raw_request["return_dict_in_generate"] = True
#     raw_request["output_scores"] = True
#     top_k_per_token: int = raw_request["top_k_per_token"]
#     del raw_request["top_k_per_token"]
#     #-----------------------------------------
#     # getting index of last question
#     index_in_prompt = raw_request["prompt"].rfind("Question")
#     tokens_question = tokenizer(raw_request["prompt"][index_in_prompt:], return_tensors="pt", return_token_type_ids=False)
#     len_tokens_question = tokens_question["input_ids"].shape[1]
#     #-----------------------------------------
    
#     if len(raw_request["stop_sequences"]) > 0:
#         stop_sequence_ids = tokenizer(
#             raw_request["stop_sequences"], return_token_type_ids=False, add_special_tokens=False
#         )
#         assert len(stop_sequence_ids.input_ids) == 1, "Total number of stop words should be 1."
#         assert len(stop_sequence_ids.input_ids[0]) == 1, "Total number of tokens in each stop word should be 1."
#         del raw_request["stop_sequences"]
#         raw_request["eos_token_id"] = stop_sequence_ids.input_ids[0][0]

#     # Strip out irrelevant parameters
#     relevant_raw_request = {
#         key: raw_request[key]
#         for key in raw_request
#         if key not in ["engine", "prompt", "echo_prompt", "stop_sequences"]
#     }

#     # TODO: using GenerationConfig
#     #-----------------------------------------
#     # Use HuggingFace's `generate` method.
#     output = model.generate(**encoded_input, **relevant_raw_request)
#     sequences = output.sequences
#     scores = output.scores
    
#     #storing hidden states
#     if raw_request["output_hidden_states"]:
#         # instance_hiddenstates = {"len_tokens_question": len_tokens_question, "hidden_states" : torch.cat(output.hidden_states[0])}
#         # hidden_states = hidden_states_process(instance_hiddenstates)
#         #print(f'{"x"*100}\n{hidden_states["sum"].shape=}\n{hidden_states["last"].shape=}\n{"x"*100}')
#         # del instance_hiddenstates

#         hs = torch.cat(output.hidden_states[0][-len_tokens_question:]).detach().cpu().numpy()
#         # cropped_hidden_states = [copy.deepcopy(i[-len_tokens_question:,:].mean(0).detach().cpu().numpy()) for i in output.hidden_states[0]]
#         # # for i in output.hidden_states[0]:
#         # #     cropped_hidden_states.append(copy.deepcopy(i[-len_tokens_question:,:].mean(0).detach().cpu().numpy()))
#         # hidden_states = cropped_hidden_states
#         # del output
#         #hidden_states =reduce(copy.deepcopy(hs[:,-len_tokens_question:,:]), "l s d -> l d", "mean")
#         hidden_states = {"last": copy.deepcopy(hs[:,-1,:]), "sum":reduce(copy.deepcopy(hs[:,-len_tokens_question:,:]), "l s d -> l d", "mean")}
#     else:
#         hidden_states = None

#     # Compute logprobs for each completed sequence.
#     all_logprobs_of_chosen_tokens = []
#     all_top_logprobs_dicts = []
#     for completion_id in range(raw_request["num_return_sequences"]):
#         logprobs_of_chosen_tokens = []
#         top_logprobs_dicts = []
#         for i in range(len(sequences[completion_id]) - len(encoded_input.input_ids[0])):
#             logprobs = torch.nn.functional.log_softmax(scores[i][completion_id], dim=0)

#             # Get top tokens in terms of log probability.
#             topk_logprobs = torch.topk(logprobs, k=top_k_per_token)
#             top_logprobs_dicts.append(
#                 {
#                     tokenizer.convert_ids_to_tokens(k.item()): v.item()
#                     for (k, v) in zip(topk_logprobs.indices, topk_logprobs.values)
#                 }
#             )

#             # Get log probability of chosen token.
#             j = i + len(encoded_input.input_ids[0])
#             logprobs_of_chosen_tokens.append(logprobs[sequences[completion_id][j]].item())
#         all_logprobs_of_chosen_tokens.append(logprobs_of_chosen_tokens)
#         all_top_logprobs_dicts.append(top_logprobs_dicts)

#     # Remove prompt from the start of each sequence if echo_prompt is False.
#     if not raw_request["echo_prompt"]:
#         sequences = [sequence[len(encoded_input.input_ids[0]) :] for sequence in sequences]

#     all_tokens = [[tokenizer.decode(token) for token in sequence_tokens] for sequence_tokens in sequences]
#     all_decoded_text = tokenizer.batch_decode(sequences)

#     completions = []
#     for decoded_text, tokens, logprobs_of_chosen_tokens, top_logprobs_dicts in zip(
#         all_decoded_text, all_tokens, all_logprobs_of_chosen_tokens, all_top_logprobs_dicts
#     ):
#         completions.append(
#             {
#                 "text": decoded_text,
#                 "tokens": tokens,
#                 "logprobs": logprobs_of_chosen_tokens,
#                 "top_logprobs_dicts": top_logprobs_dicts,
#                 "hidden_states": hidden_states
#             }
#         )
#     torch.cuda.empty_cache()
#     return {"completions": completions, "input_length": len(encoded_input.input_ids[0])}
