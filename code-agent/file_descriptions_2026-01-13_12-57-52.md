# File Descriptions

Generated on: 2026-01-13 16:43:17

---

## ./camel/__init__.py

Initialization module for the CAMEL package that manages logging functions and version information.

---

## ./camel/agents/__init__.py

This file initializes the agents package by importing and exposing various agent classes and tool agents for external use.

---

## ./camel/agents/_types.py

Defines Pydantic data models for tool call requests and model responses used in the CAMEL AI agent framework.

---

## ./camel/agents/_utils.py

Utility functions for building prompts, extracting tool calls, and safely serializing models used by CAMEL agents.

---

## ./camel/agents/base.py

Defines an abstract base class for CAMEL agents with reset and step methods to be implemented by subclasses.

---

## ./camel/agents/chat_agent.py

Defines a chat agent class with utilities for managing streaming responses, tool interactions, memory, and integration with various AI model backends and logging frameworks.

---

## ./camel/agents/critic_agent.py

Defines a CriticAgent class that evaluates and selects options from proposals using a chat-based AI model with configurable retry and verbosity settings.

---

## ./camel/agents/deductive_reasoner_agent.py

Defines a DeductiveReasonerAgent class that performs deductive reasoning to derive conditions and quality assessments for transitioning from a starting state to a target state using a specified reasoning model.

---

## ./camel/agents/embodied_agent.py

Defines the EmbodiedAgent class for managing conversations with CAMEL embodied agents, integrating tool agents and code interpreters for enhanced interaction capabilities.

---

## ./camel/agents/knowledge_graph_agent.py

Defines KnowledgeGraphAgent, a chat agent that extracts and structures nodes and relationships from unstructured content into graph elements using a language model.

---

## ./camel/agents/mcp_agent.py

Defines MCPAgent, a specialized ChatAgent that integrates and utilizes tools from MCP registries to enhance search and problem-solving capabilities through tool-assisted responses.

---

## ./camel/agents/multi_hop_generator_agent.py

Defines a programmable chat agent that generates complex multi-hop question-answer pairs requiring multi-step reasoning from a given context.

---

## ./camel/agents/programmed_agent_instruction.py

Defines an abstract programmable chat agent with atomic operation support, state repair mechanisms, and a decorator for ensuring atomic execution of agent instructions.

---

## ./camel/agents/repo_agent.py

Defines RepoAgent, a specialized chat agent that integrates GitHub repository content for code generation using full context or retrieval-augmented generation modes.

---

## ./camel/agents/role_assignment_agent.py

Defines a RoleAssignmentAgent class that generates and assigns expert role names and descriptions based on a given task prompt using a language model.

---

## ./camel/agents/search_agent.py

Defines a SearchAgent class that summarizes text based on a query and evaluates the relevance of answers using a chat-based model backend.

---

## ./camel/agents/task_agent.py

Defines TaskSpecifyAgent and TaskPlannerAgent classes for specifying task details and dividing tasks into subtasks using AI-driven chat agents.

---

## ./camel/agents/tool_agents/__init__.py

Initialization module for the tool_agents package, exposing BaseToolAgent and HuggingFaceToolAgent classes.

---

## ./camel/agents/tool_agents/base.py

Defines a base class for tool agents with name, description, and placeholder methods for reset and step actions.

---

## ./camel/agents/tool_agents/hugging_face_tool_agent.py

Defines a HuggingFaceToolAgent class that wraps HuggingFace transformers tool agents to perform diverse tasks such as question answering, image captioning, speech-to-text, text-to-image generation, and more.

---

## ./camel/benchmarks/__init__.py

This file initializes the benchmarks package by importing and exposing various benchmark classes and utilities for evaluation purposes.

---

## ./camel/benchmarks/apibank.py

Defines the APIBankBenchmark class for downloading, loading, and processing the API-Bank dataset to evaluate tool-augmented large language models.

---

## ./camel/benchmarks/apibench.py

Defines the APIBench benchmark for evaluating language models on API-related tasks using datasets from Huggingface, TensorFlow Hub, and TorchHub with oracle retrieval.

---

## ./camel/benchmarks/base.py

Defines an abstract base class for benchmarks that manages data loading, downloading, running evaluations with chat agents, and storing results.

---

## ./camel/benchmarks/browsecomp.py

Defines data models and evaluation logic for structured query responses, grading, and benchmark result reporting within the CAMEL AI benchmarking framework.

---

## ./camel/benchmarks/gaia.py

Defines the GAIA benchmark for evaluating general AI assistants, including a retriever protocol, a default retriever implementation, and benchmark management with data handling and retrieval capabilities.

---

## ./camel/benchmarks/mock_website/mock_web.py

Python script to download, set up, and run mock website projects with logging and subprocess output handling.

---

## ./camel/benchmarks/mock_website/shopping_mall/app.py

A Flask application module for a mock shopping mall website that manages product loading, shopping cart operations, logging setup, and task completion verification based on a ground truth cart.

---

## ./camel/benchmarks/nexus.py

Defines the NexusBenchmark class for downloading, loading, and evaluating the Nexus function calling benchmark datasets with support for multiple API benchmarks.

---

## ./camel/benchmarks/ragbench.py

Defines the RAGAS benchmark for evaluating retrieval-augmented generation by annotating datasets, computing metrics like RMSE and AUROC, and assessing context relevance and faithfulness.

---

## ./camel/bots/__init__.py

Initialization module that imports and exposes various chatbot classes and related Slack event models for the CAMEL AI project.

---

## ./camel/bots/discord/__init__.py

Initialization module for the Discord bot package, exposing core classes and installation stores.

---

## ./camel/bots/discord/discord_app.py

Defines a Discord bot application class that manages bot authentication, message handling, and optional OAuth installation flow using discord.py and FastAPI.

---

## ./camel/bots/discord/discord_installation.py

Defines a class representing a Discord application's installation details, including authentication tokens and installation timestamps for a specific guild.

---

## ./camel/bots/discord/discord_store.py

Defines abstract and SQLite-based classes for storing, retrieving, and managing Discord installation data asynchronously.

---

## ./camel/bots/slack/__init__.py

Initialization module for the Slack bot package, exposing key classes and data models for Slack event handling and authentication.

---

## ./camel/bots/slack/models.py

Defines Pydantic data models representing Slack event structures, including authorization profiles, event details, and app mention events for use in Slack bot integrations.

---

## ./camel/bots/slack/slack_app.py

Defines a SlackApp class that initializes and manages an asynchronous Slack Bolt app with event handling and OAuth support.

---

## ./camel/bots/telegram_bot.py

Defines a Telegram bot class that integrates with a ChatAgent to handle and respond to user messages using the telebot library.

---

## ./camel/configs/__init__.py

This file aggregates and exposes configuration classes and API parameter constants for various AI service providers within the CAMEL project.

---

## ./camel/configs/aihubmix_config.py

Defines configuration parameters for generating chat completions using the AihubMix API.

---

## ./camel/configs/aiml_config.py

Defines a Pydantic configuration class for specifying parameters used in generating chat completions with the AIML API.

---

## ./camel/configs/amd_config.py

Defines the AMDConfig class for configuring AMD API language model parameters such as temperature, penalties, max tokens, and response streaming options.

---

## ./camel/configs/anthropic_config.py

Defines a configuration class for specifying parameters used in generating chat completions with the Anthropic API.

---

## ./camel/configs/base_config.py

Defines an abstract base configuration class for models that validates and manages a list of function tools and supports conversion to a dictionary format.

---

## ./camel/configs/bedrock_config.py

Defines a configuration class for specifying parameters used in generating chat completions with OpenAI-compatible Bedrock models.

---

## ./camel/configs/cerebras_config.py

Defines a configuration class for setting parameters to generate chat completions compatible with Cerebras AI models.

---

## ./camel/configs/cohere_config.py

Defines a configuration class for setting parameters used in generating chat completions with the Cohere API.

---

## ./camel/configs/cometapi_config.py

Defines a configuration class for setting parameters to generate chat completions using CometAPI's OpenAI-compatible API.

---

## ./camel/configs/crynux_config.py

Defines a configuration class for setting parameters used in generating chat completions with the OpenAI API.

---

## ./camel/configs/deepseek_config.py

Defines a Pydantic configuration model for specifying parameters to generate chat completions using the DeepSeek API.

---

## ./camel/configs/function_gemma_config.py

Defines configuration parameters for generating completions using FunctionGemma via Ollama's native API with custom function calling templates.

---

## ./camel/configs/gemini_config.py

Defines a configuration class for specifying parameters used in generating chat completions with the Gemini API.

---

## ./camel/configs/groq_config.py

Defines a configuration class for generating chat completions compatible with Groq's OpenAI API, including parameters for temperature, token limits, streaming, and tool usage.

---

## ./camel/configs/internlm_config.py

Defines the configuration parameters for generating chat completions using the InternLM API within the CAMEL framework.

---

## ./camel/configs/litellm_config.py

Defines the configuration parameters for generating chat completions using the LiteLLM API.

---

## ./camel/configs/lmstudio_config.py

Defines a configuration class for generating chat completions with OpenAI-compatible parameters, including temperature, top_p, response format, streaming, stop sequences, token limits, penalties, and tool usage options.

---

## ./camel/configs/minimax_config.py

Defines a configuration class for setting parameters to generate chat completions compatible with the Minimax API.

---

## ./camel/configs/mistral_config.py

Defines a Pydantic configuration class for specifying parameters used in generating chat completions with the Mistral API.

---

## ./camel/configs/modelscope_config.py

Defines a configuration class for specifying parameters to generate chat completions using the ModelScope API.

---

## ./camel/configs/moonshot_config.py

Defines the configuration parameters for generating chat completions using the Moonshot API.

---

## ./camel/configs/nebius_config.py

Defines the NebiusConfig class for configuring chat completion parameters compatible with Nebius AI Studio's OpenAI-like API.

---

## ./camel/configs/netmind_config.py

Defines a Pydantic configuration class for specifying parameters used in generating chat completions with OpenAI-compatible Netmind API.

---

## ./camel/configs/novita_config.py

Defines the NovitaConfig class for configuring parameters used in generating chat completions with the OpenAI API.

---

## ./camel/configs/nvidia_config.py

Defines a Pydantic configuration class for setting parameters of NVIDIA API language models, including sampling, penalties, token limits, and tool options.

---

## ./camel/configs/ollama_config.py

Defines a Pydantic-based configuration class for specifying parameters used in generating chat completions with Ollama's OpenAI-compatible API.

---

## ./camel/configs/openai_config.py

Defines a configuration class for specifying parameters used in generating chat completions with the OpenAI API.

---

## ./camel/configs/openrouter_config.py

Defines a configuration class for setting parameters to generate chat completions compatible with the OpenRouter API.

---

## ./camel/configs/ppio_config.py

Defines a Pydantic configuration model for specifying parameters used in generating chat completions with the OpenAI API.

---

## ./camel/configs/qianfan_config.py

Defines a Pydantic configuration class for specifying parameters to generate chat completions compatible with the Qianfan API.

---

## ./camel/configs/qwen_config.py

Defines the configuration parameters for generating chat completions using the Qwen API in the CAMEL framework.

---

## ./camel/configs/reka_config.py

Defines the configuration parameters for generating chat completions using the Reka API.

---

## ./camel/configs/samba_config.py

Defines configuration classes for setting parameters of SambaVerse and SambaCloud API chat completion requests.

---

## ./camel/configs/sglang_config.py

Defines a configuration class for setting parameters used in generating chat completions with the OpenAI API, including options for sampling, penalties, streaming, and tool integration.

---

## ./camel/configs/siliconflow_config.py

Defines a configuration class for specifying parameters used in generating chat completions with the SiliconFlow API.

---

## ./camel/configs/togetherai_config.py

Defines a Pydantic configuration class for specifying parameters used in generating chat completions with the TogetherAI OpenAI-compatible API.

---

## ./camel/configs/vllm_config.py

Defines a Pydantic configuration class for specifying parameters used in generating chat completions with the VLLM OpenAI-compatible API.

---

## ./camel/configs/watsonx_config.py

Defines a configuration class for specifying parameters used in generating chat completions with the IBM WatsonX API.

---

## ./camel/configs/yi_config.py

Defines the YiConfig class for configuring parameters used in generating chat completions via the Yi API, including options for tool usage, token limits, randomness, temperature, and streaming.

---

## ./camel/configs/zhipuai_config.py

Defines configuration parameters for generating chat completions using the ZhipuAI model with OpenAI-compatible API options.

---

## ./camel/data_collectors/__init__.py

Initialization module that imports and exposes data collector classes for Alpaca and ShareGPT datasets.

---

## ./camel/data_collectors/alpaca_collector.py

Defines AlpacaDataCollector, a class for collecting, filtering, and converting chat data from Alpaca agents into structured JSON format, optionally using an LLM schema converter.

---

## ./camel/data_collectors/base.py

Defines base classes and data structures for collecting and recording chat message data from agents with roles and optional function calls.

---

## ./camel/data_collectors/sharegpt_collector.py

Defines a data collector class that records and converts ShareGPT conversation data into a structured JSON format including system messages, tools, and dialogue exchanges.

---

## ./camel/datagen/__init__.py

Initialization module for the datagen package that imports and exposes CoTDataGenerator, SelfInstructPipeline, and SelfImprovingCoTPipeline classes.

---

## ./camel/datagen/cot_datagen.py

Implements a Chain of Thought data generator using chat agents with Monte Carlo Tree Search, error detection, and dual-agent verification for producing and validating reasoning paths.

---

## ./camel/datagen/evol_instruct/__init__.py

Initialization module for the evol_instruct package, exposing EvolInstructPipeline and MathEvolInstructTemplates.

---

## ./camel/datagen/evol_instruct/evol_instruct.py

Defines the EvolInstructPipeline class for iteratively evolving prompts using customizable evolution strategies and language model interactions.

---

## ./camel/datagen/evol_instruct/scorer.py

Defines abstract and concrete scorer classes using AI agents to evaluate and score the quality and characteristics of math and general problems based on diversity, difficulty, solvability, complexity, and validity.

---

## ./camel/datagen/evol_instruct/templates.py

Defines abstract and concrete classes for evolution instruction prompt templates used to generate and transform complex, human-like prompts for language model training.

---

## ./camel/datagen/self_improving_cot.py

Implements a self-improving chain-of-thought reasoning pipeline that iteratively generates, evaluates, and refines reasoning traces using chat agents and reward models.

---

## ./camel/datagen/self_instruct/__init__.py

Initialization module for the self-instruct data generation package, exposing filters and the SelfInstructPipeline class.

---

## ./camel/datagen/self_instruct/filter/__init__.py

Initialization module that imports and exposes various filtering classes and a filter registry for self-instruct data generation.

---

## ./camel/datagen/self_instruct/filter/filter_function.py

Defines abstract and concrete filter classes to evaluate and filter instructions based on criteria like length, keywords, punctuation, language, and similarity using ROUGE scores.

---

## ./camel/datagen/self_instruct/filter/filter_registry.py

Defines and manages a registry of filter constructors for creating various text filtering functions used in data generation.

---

## ./camel/datagen/self_instruct/filter/instruction_filter.py

Defines the InstructionFilter class to apply configurable filter functions for validating instructions based on various criteria.

---

## ./camel/datagen/self_instruct/self_instruct.py

Defines a pipeline for generating, filtering, and managing machine-generated instructions by combining human-written and machine-generated task samples using a ChatAgent.

---

## ./camel/datagen/self_instruct/templates.py

Defines a dataclass containing prompt templates for self-instruct data generation, specifically for classification task identification and example generation.

---

## ./camel/datagen/source2synth/__init__.py

Initialization module that imports and exposes data processing classes and models for source-to-synthetic data generation in the CAMEL project.

---

## ./camel/datagen/source2synth/data_processor.py

Defines a UserDataProcessor class that generates multi-hop question-answer pairs from user text data using configurable AI or rule-based methods, managing the full pipeline from text input to curated dataset output.

---

## ./camel/datagen/source2synth/models.py

Defines Pydantic data models for multi-hop question-answering tasks, including reasoning steps, QA pairs, and context prompts.

---

## ./camel/datagen/source2synth/user_data_processor_config.py

Defines a Pydantic configuration model for user data processing parameters, including dataset size, text length limits, complexity threshold, random seed, AI model usage, and a multi-hop text generation agent.

---

## ./camel/datahubs/__init__.py

Initialization module for the datahubs package that imports and exposes dataset manager classes and a record model.

---

## ./camel/datahubs/base.py

Abstract base class defining the interface for managing datasets and their records, including creation, listing, updating, and deletion operations.

---

## ./camel/datahubs/huggingface.py

A dataset manager class for creating, updating, and managing datasets and dataset cards on the Hugging Face Hub using the Hugging Face API.

---

## ./camel/datahubs/models.py

Defines a Pydantic data model for a Record with optional id, metadata, and content fields allowing extra attributes.

---

## ./camel/datasets/__init__.py

Initialization file for the camel.datasets package that imports and exposes dataset generator classes and data models.

---

## ./camel/datasets/base_generator.py

Defines an abstract base class for asynchronously generating, iterating, and optionally caching synthetic data points in a dataset.

---

## ./camel/datasets/few_shot_generator.py

Defines a FewShotGenerator class that creates synthetic data points using few-shot prompting with a seed dataset, an LLM-based agent, and a verifier to ensure logical consistency.

---

## ./camel/datasets/models.py

Defines a Pydantic data model for a dataset data point including question, final answer, optional rationale, and metadata with serialization methods.

---

## ./camel/datasets/self_instruct_generator.py

Defines a SelfInstructGenerator class that creates synthetic question-and-answer datapoints by generating new instructions, corresponding rationale code, and verified answers using seed data and AI agents.

---

## ./camel/datasets/static_dataset.py

Defines a PyTorch-compatible static dataset class that loads, validates, and standardizes data from various sources into a list of DataPoint instances.

---

## ./camel/embeddings/__init__.py

This file initializes the embeddings module by importing and exposing various embedding classes for use within the CAMEL-AI project.

---

## ./camel/embeddings/azure_embedding.py

Defines the AzureEmbedding class for generating text embeddings using Azure's OpenAI models with configurable parameters and API integration.

---

## ./camel/embeddings/base.py

Abstract base class defining the interface for generating text embeddings and retrieving their dimensionality.

---

## ./camel/embeddings/gemini_embedding.py

Defines a GeminiEmbedding class for generating text embeddings using Google's Gemini models with configurable model types, API keys, and task-specific optimizations.

---

## ./camel/embeddings/jina_embedding.py

Defines a JinaEmbedding class that generates text and image embeddings using Jina AI's API with support for various model types, embedding formats, and tasks.

---

## ./camel/embeddings/mistral_embedding.py

Defines a MistralEmbedding class for generating text embeddings using Mistral's API with configurable model types and output dimensions.

---

## ./camel/embeddings/openai_compatible_embedding.py

Defines an OpenAI-compatible text embedding class that generates and manages embedding vectors using a specified model and API credentials.

---

## ./camel/embeddings/openai_embedding.py

Defines the OpenAIEmbedding class to generate text embeddings using OpenAI's embedding models with configurable parameters and API integration.

---

## ./camel/embeddings/sentence_transformers_embeddings.py

This file defines a class for generating normalized text embeddings using Sentence Transformers with customizable models.

---

## ./camel/embeddings/together_embedding.py

Defines a TogetherEmbedding class that generates text embeddings using Together AI's models via their API.

---

## ./camel/embeddings/vlm_embedding.py

Defines a VisionLanguageEmbedding class that generates image and text embeddings using a specified multimodal model like OpenAI's CLIP.

---

## ./camel/environments/__init__.py

This file initializes the environments package by importing and exposing various environment classes and related components for reinforcement learning tasks.

---

## ./camel/environments/models.py

Defines data models and an asynchronous protocol for actions, observations, step results, and environment interactions in a reinforcement learning framework.

---

## ./camel/environments/multi_step.py

Defines an abstract multi-step reinforcement learning environment class for interacting with large language models, managing state, actions, observations, and lifecycle methods.

---

## ./camel/environments/rlcards_env.py

Defines a wrapper environment integrating RLCard games with CAMEL for reinforcement learning using LLMs, including action extraction and multi-agent setup.

---

## ./camel/environments/single_step.py

Defines a single-step reinforcement learning environment for LLM agents that samples problems from a dataset, obtains one response per problem, verifies correctness, and computes rewards accordingly.

---

## ./camel/environments/tic_tac_toe.py

Defines a Tic Tac Toe environment with AI opponent strategies including an optimal minimax algorithm and a move extraction method from text.

---

## ./camel/extractors/__init__.py

Initialization module that imports and exposes base extractor classes and various Python data structure extraction strategies.

---

## ./camel/extractors/base.py

Defines abstract base classes and a configurable asynchronous multi-stage pipeline for text extraction strategies with caching, batching, and resource monitoring support.

---

## ./camel/extractors/python_strategies.py

Defines Python extraction strategies to parse and normalize content from LaTeX \boxed{} environments, Python lists, and dictionaries.

---

## ./camel/generators.py

Defines the SystemMessageGenerator class for creating role-based system messages using customizable prompts and metadata validation.

---

## ./camel/human.py

Defines a Human class that interacts with users by displaying message options, receiving input, and processing their choices in a conversational interface.

---

## ./camel/interpreters/__init__.py

This file initializes the interpreters module by importing and exposing various interpreter classes and related error handling.

---

## ./camel/interpreters/base.py

Defines an abstract base class for code interpreters specifying methods for running code, supported code types, updating action spaces, and executing commands.

---

## ./camel/interpreters/docker_interpreter.py

Defines a DockerInterpreter class that executes code snippets or files in various languages inside a Docker container, managing container lifecycle, code execution, and output handling.

---

## ./camel/interpreters/e2b_interpreter.py

Python class implementing an E2B code interpreter that executes various code types in a sandboxed environment with optional user confirmation and API key authentication.

---

## ./camel/interpreters/internal_python_interpreter.py

Defines a secure, customizable Python interpreter class that executes LLM-generated code with controlled action spaces, import whitelists, and fuzzy variable matching.

---

## ./camel/interpreters/interpreter_error.py

Defines a custom InterpreterError exception for handling errors that can be resolved by regenerating code.

---

## ./camel/interpreters/ipython_interpreter.py

Defines a JupyterKernelInterpreter class for securely executing and capturing output from code strings in a Jupyter kernel environment.

---

## ./camel/interpreters/microsandbox_interpreter.py

Defines a MicrosandboxInterpreter class for securely executing Python, JavaScript, and shell code using a self-hosted microsandbox platform with configurable server, authentication, and execution settings.

---

## ./camel/interpreters/subprocess_interpreter.py

Defines SubprocessInterpreter, a class for securely executing Python, Bash, and R code files or strings in subprocesses with output capture and optional user confirmation.

---

## ./camel/loaders/__init__.py

This file initializes the camel.loaders package by importing and exposing various data loader classes and functions, while handling deprecated attribute access.

---

## ./camel/loaders/apify_reader.py

This file defines an Apify class for interacting with the Apify platform, enabling running actors and accessing datasets via the Apify API.

---

## ./camel/loaders/base_io.py

Defines an abstract base class and utility functions for loading and representing various file types (e.g., DOCX, PDF, TXT, JSON, HTML) as File objects with metadata and document content.

---

## ./camel/loaders/base_loader.py

Abstract base class defining the interface and common logic for loading data from single or multiple sources in CAMEL.

---

## ./camel/loaders/chunkr_reader.py

Defines a ChunkrReader class for submitting documents to the Chunkr API for OCR processing and retrieving the processed results asynchronously.

---

## ./camel/loaders/crawl4ai_reader.py

Defines an asynchronous web crawler class that converts websites into structured, LLM-ready data using CSS selectors or LLM-based extraction with configurable crawling depth and strategies.

---

## ./camel/loaders/firecrawl_reader.py

Python class wrapper for interacting with the Firecrawl API to crawl, scrape, and extract structured data from websites into LLM-ready markdown.

---

## ./camel/loaders/jina_url_reader.py

Defines a JinaURLReader class that fetches and returns cleaner, LLM-optimized content from URLs using the Jina AI API with configurable options.

---

## ./camel/loaders/markitdown.py

This file defines the MarkItDownLoader class, which converts various supported file types into Markdown format, optionally using an LLM client for enhanced processing.

---

## ./camel/loaders/mineru_extractor.py

Python class for interacting with the MinerU API to extract document content including OCR, formula recognition, and tables from URLs or batches of URLs.

---

## ./camel/loaders/mistral_reader.py

Defines a MistralReader class that loads documents or images, encodes files to base64, and extracts text using the Mistral OCR API.

---

## ./camel/loaders/scrapegraph_reader.py

This file defines the ScrapeGraphAI class for performing AI-powered web scraping and searching using the ScrapeGraphAI API.

---

## ./camel/loaders/unstructured_io.py

Utility class for parsing, creating, and managing unstructured document elements with support for local files and URLs, including metadata handling and cloud integration.

---

## ./camel/logger.py

This file configures and manages logging for the CAMEL library, including setting log levels, enabling/disabling logging, and adding file handlers.

---

## ./camel/memories/__init__.py

Initialization module for the CAMEL AI memories package, exposing key memory classes, context creators, and memory blocks.

---

## ./camel/memories/agent_memories.py

Defines the ChatHistoryMemory class for managing an agent's chat history with optional storage, retrieval, and cleaning of tool-related messages.

---

## ./camel/memories/base.py

Defines abstract base classes for memory blocks, context creation strategies, and agent memory management within the CAMEL AI framework.

---

## ./camel/memories/blocks/__init__.py

Initialization file that imports and exposes ChatHistoryBlock and VectorDBBlock classes for the memories.blocks module.

---

## ./camel/memories/blocks/chat_history_block.py

Defines a ChatHistoryBlock class for managing and retrieving chat history records using a key-value storage backend with optional windowed access and message retention scoring.

---

## ./camel/memories/blocks/vectordb_block.py

Defines a VectorDBBlock class for storing, retrieving, and managing chat memory records using vector embeddings and a vector database storage.

---

## ./camel/memories/context_creators/__init__.py

Initialization module for context creators, exposing the ScoreBasedContextCreator class.

---

## ./camel/memories/context_creators/score_based.py

Defines a score-based context creator that orders memory records chronologically and estimates token counts efficiently for constructing message contexts.

---

## ./camel/memories/records.py

Defines the MemoryRecord class for storing and reconstructing message records with metadata in the CAMEL memory system.

---

## ./camel/messages/__init__.py

This file initializes the camel.messages package by importing and exposing various message types, conversion utilities, and message-related classes for use in the CAMEL-AI framework.

---

## ./camel/messages/base.py

Defines the BaseMessage dataclass representing chat messages with roles, content, media attachments, and metadata for the CAMEL chat system.

---

## ./camel/messages/conversion/__init__.py

Initialization module for message conversion components including AlpacaItem, ShareGPT conversation models, and HermesFunctionFormatter.

---

## ./camel/messages/conversion/alpaca.py

Defines the AlpacaItem class for parsing, validating, and formatting instruction-response data in the Alpaca text format.

---

## ./camel/messages/conversion/conversation_models.py

Defines Pydantic models with validation for representing and validating ShareGPT conversation messages, tool calls, and tool responses in a structured format.

---

## ./camel/messages/conversion/sharegpt/__init__.py

Initialization file for the sharegpt message conversion module, exposing the HermesFunctionFormatter class.

---

## ./camel/messages/conversion/sharegpt/function_call_formatter.py

Defines an abstract base class for formatting and extracting function call and response information in message strings.

---

## ./camel/messages/conversion/sharegpt/hermes/__init__.py

Initialization module for the Hermes message conversion package, exposing the HermesFunctionFormatter class.

---

## ./camel/messages/conversion/sharegpt/hermes/hermes_function_formatter.py

Defines Hermes-style function call and response classes with methods to parse and format tool calls and responses in a specific tagged JSON string format.

---

## ./camel/messages/func_message.py

Defines the FunctionCallingMessage class for handling function-related messages and converting them to various OpenAI and ShareGPT message formats.

---

## ./camel/models/__init__.py

This file initializes the camel.models package by importing and exposing various AI model classes and related utilities for easy access.

---

## ./camel/models/_utils.py

Utility functions for modifying OpenAIMessage content to enforce response formatting based on a given Pydantic model schema.

---

## ./camel/models/aihubmix_model.py

Defines the AihubMixModel class that integrates the AihubMix API with an OpenAI-compatible interface for configurable AI model interactions.

---

## ./camel/models/aiml_model.py

Defines the AIMLModel class that provides a unified OpenAI-compatible interface for interacting with the AIML API, including configuration, authentication, and request handling.

---

## ./camel/models/amd_model.py

Defines the AMDModel class that provides a unified OpenAI-compatible interface for interacting with the AMD API, including configuration validation and API key management.

---

## ./camel/models/anthropic_model.py

Defines the AnthropicModel class providing an OpenAI-compatible interface to the Anthropic API, including message preprocessing and configuration handling.

---

## ./camel/models/aws_bedrock_model.py

Defines the AWSBedrockModel class that provides an OpenAI-compatible interface for interacting with the AWS Bedrock API.

---

## ./camel/models/azure_openai_model.py

Defines the AzureOpenAIModel class that provides a unified interface for interacting with Azure OpenAI API, supporting synchronous and asynchronous clients, token counting, and configurable authentication options.

---

## ./camel/models/base_audio_model.py

Abstract base class defining the interface and common functionality for Text-to-Speech and Speech-to-Text audio models.

---

## ./camel/models/base_model.py

Defines abstract base classes and stream wrapper utilities for handling and logging synchronous and asynchronous chat completion streams in the CAMEL AI framework.

---

## ./camel/models/cerebras_model.py

Defines the CerebrasModel class that provides an OpenAI-compatible interface to interact with the Cerebras LLM API, including configuration, authentication, and request handling.

---

## ./camel/models/cohere_model.py

Defines a CohereModel class that provides a unified interface to interact with the Cohere API for chat-based language models within the CAMEL framework.

---

## ./camel/models/cometapi_model.py

Defines the CometAPIModel class, an OpenAI-compatible interface for interacting with the CometAPI LLM service, including configuration, authentication, and request handling.

---

## ./camel/models/crynux_model.py

Defines the CrynuxModel class, an OpenAI-compatible backend client for interacting with the Crynux AI service, including configuration, authentication, and request handling.

---

## ./camel/models/deepseek_model.py

Defines the DeepSeekModel class that integrates the DeepSeek API with an OpenAI-compatible interface for chat-based AI models.

---

## ./camel/models/fish_audio_model.py

Defines the FishAudioModel class for interfacing with FishAudio's Text-to-Speech and Speech-to-Text API services, including methods for converting text to speech and speech to text.

---

## ./camel/models/function_gemma_model.py

Defines the FunctionGemmaModel backend for the Ollama platform, enabling specialized function calling with custom chat template formatting and conversion between CAMEL's OpenAI-style tool schemas and FunctionGemma's native format.

---

## ./camel/models/gemini_model.py

Defines the GeminiModel class that provides a unified OpenAI-compatible interface for interacting with the Gemini API, including message processing and API call management.

---

## ./camel/models/groq_model.py

Defines the GroqModel class that provides a Groq-served LLM API with an OpenAI-compatible interface, including configuration, authentication, and request handling.

---

## ./camel/models/internlm_model.py

Defines the InternLMModel class providing a unified OpenAI-compatible interface for interacting with the InternLM API, including configuration, authentication, and synchronous inference support.

---

## ./camel/models/litellm_model.py

Defines the LiteLLMModel class that provides an OpenAI-compatible backend interface for the LiteLLM language model, including response conversion and configuration handling.

---

## ./camel/models/lmstudio_model.py

Defines the LMStudioModel class that adapts an LMStudio-served language model to the OpenAICompatibleModel interface with configurable parameters.

---

## ./camel/models/minimax_model.py

Defines the MinimaxModel class, an OpenAI-compatible interface for interacting with the Minimax LLM API, including configuration, authentication, and request handling.

---

## ./camel/models/mistral_model.py

Defines the MistralModel class that integrates the Mistral API into a unified BaseModelBackend interface for chat completion with configurable settings and token counting.

---

## ./camel/models/model_factory.py

Defines a factory class that maps model platform types to their corresponding backend model classes for instantiating AI models.

---

## ./camel/models/model_manager.py

Manages multiple AI model backends by selecting and scheduling models according to configurable strategies such as round-robin.

---

## ./camel/models/modelscope_model.py

Defines a ModelScopeModel class that integrates ModelScope API with an OpenAI-compatible interface for unified model interaction.

---

## ./camel/models/moonshot_model.py

Defines the MoonshotModel class that provides a unified OpenAI-compatible interface to interact with the Moonshot API for chat-based language models.

---

## ./camel/models/nebius_model.py

Defines the NebiusModel class, an OpenAI-compatible interface for accessing the Nebius AI Studio LLM API with configurable parameters and authentication.

---

## ./camel/models/nemotron_model.py

Defines the NemotronModel class as an OpenAI-compatible API backend for Nvidia's Nemotron model, handling authentication, configuration, and API interaction without token counting support.

---

## ./camel/models/netmind_model.py

Defines the NetmindModel class, an OpenAI-compatible backend client for interacting with Netmind's AI models via their API.

---

## ./camel/models/novita_model.py

Defines the NovitaModel class, an OpenAI-compatible backend client for interacting with Novita AI models, including configuration, authentication, and API request handling.

---

## ./camel/models/nvidia_model.py

Defines the NvidiaModel class that provides a unified OpenAI-compatible interface for interacting with NVIDIA's API, including configuration, authentication, and request handling.

---

## ./camel/models/ollama_model.py

Defines the OllamaModel class that interfaces with the Ollama service, managing model configuration, server startup, and API communication compatible with OpenAI's API.

---

## ./camel/models/openai_audio_models.py

Defines a class to interact with OpenAI's Text-to-Speech and Speech-to-Text models, enabling conversion between text and audio with support for chunked processing and file storage.

---

## ./camel/models/openai_compatible_model.py

Defines the OpenAICompatibleModel class that provides a backend for models supporting OpenAI-compatible API interactions with configurable clients, token counting, and retry logic.

---

## ./camel/models/openai_model.py

Defines the OpenAIModel class that provides a unified interface for interacting with OpenAI's API, supporting synchronous and asynchronous clients, configuration, token counting, and API key management.

---

## ./camel/models/openrouter_model.py

Defines the OpenRouterModel class that integrates OpenRouter's LLM API with a unified OpenAI-compatible interface, handling configuration, authentication, and API interaction.

---

## ./camel/models/ppio_model.py

Defines the PPIOModel class, an OpenAI-compatible backend client for interacting with the PPIO API for language models.

---

## ./camel/models/qianfan_model.py

Defines the QianfanModel class, an OpenAI-compatible backend for interacting with Baidu's Qianfan AI service, including configuration, authentication, and API request handling.

---

## ./camel/models/qwen_model.py

Defines the QwenModel class that provides a unified OpenAI-compatible interface for interacting with the Qwen API, including configuration, authentication, and request handling.

---

## ./camel/models/reka_model.py

Defines the RekaModel class that integrates the Reka API with an OpenAI-compatible interface for chat-based language model interactions.

---

## ./camel/models/reward/__init__.py

This file initializes the reward models package by importing and exposing key reward model classes and the evaluator.

---

## ./camel/models/reward/base_reward_model.py

Abstract base class defining the interface for reward models that evaluate messages and return scoring metrics.

---

## ./camel/models/reward/evaluator.py

Defines an Evaluator class that uses a reward model to score and filter messages based on specified thresholds.

---

## ./camel/models/reward/nemotron_model.py

Defines the NemotronRewardModel class for evaluating chat messages using the Nemotron model with OpenAI-compatible API integration.

---

## ./camel/models/reward/skywork_model.py

Defines the SkyworkRewardModel class that loads a transformer-based reward model from Huggingface to evaluate and score chat messages.

---

## ./camel/models/samba_model.py

Defines the SambaModel class to interface with SambaNova AI services, supporting synchronous and asynchronous API calls with configurable options for model type, authentication, and request handling.

---

## ./camel/models/sglang_model.py

Defines the SGLangModel class as an interface for interacting with the SGLang language model service, including configuration, client management, and API call handling.

---

## ./camel/models/siliconflow_model.py

Defines the SiliconFlowModel class that integrates the SiliconFlow API with a unified OpenAI-compatible interface for language model interactions.

---

## ./camel/models/stub_model.py

Defines a stub model and token counter for testing purposes that simulate model behavior with fixed responses and token counts.

---

## ./camel/models/togetherai_model.py

Implements the TogetherAIModel class as an OpenAI-compatible backend for interacting with Together AI's chat models using configurable API settings.

---

## ./camel/models/vllm_model.py

Defines a Python class for interfacing with the vLLM model service, including automatic server startup and OpenAI-compatible API calls.

---

## ./camel/models/volcano_model.py

Defines the VolcanoModel class that integrates the Volcano Engine API with an OpenAI-compatible interface for chat-based AI models.

---

## ./camel/models/watsonx_model.py

Defines the WatsonXModel class that integrates IBM WatsonX API with a unified BaseModelBackend interface for chat-based AI model interactions.

---

## ./camel/models/yi_model.py

Defines the YiModel class that provides a unified OpenAI-compatible interface for interacting with the Yi API, including configuration, authentication, and request handling.

---

## ./camel/models/zhipuai_model.py

Defines the ZhipuAIModel class that provides an OpenAI-compatible interface for interacting with the ZhipuAI API, including synchronous and asynchronous chat completion requests.

---

## ./camel/parsers/__init__.py

Initialization module for the CAMEL project parsers, exposing the extract_tool_calls_from_text function.

---

## ./camel/parsers/mcp_tool_call_parser.py

Utility functions to extract and parse JSON or JSON-like tool call data from model-generated text outputs.

---

## ./camel/personas/__init__.py

Initialization module for the personas package, exposing Persona and PersonaHub classes.

---

## ./camel/personas/persona.py

Defines a Pydantic-based Persona class representing a character with attributes and prompts for text-to-persona and persona-to-persona interactions, including JSON serialization support.

---

## ./camel/personas/persona_hub.py

Defines the PersonaHub class for managing and generating diverse synthetic personas using large language models to facilitate persona-driven data synthesis.

---

## ./camel/prompts/__init__.py

This file initializes the prompts package by importing and exposing various prompt template classes and dictionaries for different AI tasks.

---

## ./camel/prompts/ai_society.py

Defines a collection of structured text prompts for simulating collaborative interactions and task specifications between AI assistants and users within an AI society framework.

---

## ./camel/prompts/base.py

Defines TextPrompt and CodePrompt classes with decorators to automatically convert string returns to prompt instances and provide keyword extraction and formatting features.

---

## ./camel/prompts/code.py

Defines a collection of structured text prompts for guiding AI interactions in programming-related tasks within the CAMEL framework.

---

## ./camel/prompts/evaluation.py

Defines a dictionary of text prompts for generating evaluation questions to assess knowledge emergence in specific fields.

---

## ./camel/prompts/generate_text_embedding_data.py

Defines prompt templates for generating synthetic text embedding tasks and corresponding query, positive, and hard negative document samples based on a research paper.

---

## ./camel/prompts/image_craft.py

Defines a prompt template dictionary for an AI assistant to create original images based on descriptive captions in the ImageCraft task.

---

## ./camel/prompts/misalignment.py

Defines a dictionary of text prompts for generating and specifying malicious tasks in a misalignment scenario involving AI assistant and user roles.

---

## ./camel/prompts/multi_condition_image_craft.py

Defines a prompt template dictionary for generating assistant prompts that create images based on multiple text and image conditions.

---

## ./camel/prompts/object_recognition.py

Defines a prompt template for an assistant to perform object recognition by listing detected objects in an image as a numbered list.

---

## ./camel/prompts/persona_hub.py

Defines a PersonaHubPrompt class containing text prompts for generating and relating personas based on given text or existing persona descriptions.

---

## ./camel/prompts/prompt_templates.py

Defines a class for generating and retrieving task-specific prompt templates based on task and role types.

---

## ./camel/prompts/role_description_prompt_template.py

Defines a dictionary of text prompts for describing AI roles and their collaboration in task completion within the CAMEL AI framework.

---

## ./camel/prompts/solution_extraction.py

Defines a prompt template dictionary for extracting detailed and complete solutions from conversations between a user and an AI assistant.

---

## ./camel/prompts/task_prompt_template.py

Defines a dictionary class mapping task types to their corresponding prompt template dictionaries for various AI tasks.

---

## ./camel/prompts/translation.py

Defines a dictionary of text prompts for an AI assistant specialized in translating English text to a specified language.

---

## ./camel/prompts/video_description_prompt.py

Defines a dictionary of text prompts for video description tasks, including an assistant prompt for providing shot descriptions of video content.

---

## ./camel/responses/__init__.py

Initialization file for the responses module that imports and exposes the ChatAgentResponse class.

---

## ./camel/responses/agent_responses.py

Defines a Pydantic model representing a chat agent's response, including messages, termination status, and additional info.

---

## ./camel/retrievers/__init__.py

This file initializes the retrievers module by importing and exposing various retriever classes for information retrieval.

---

## ./camel/retrievers/auto_retriever.py

Defines the AutoRetriever class for automatic information retrieval using configurable vector storage backends and embedding models.

---

## ./camel/retrievers/base.py

Abstract base class defining the interface for information retrievers with unimplemented query and process methods to be overridden by subclasses.

---

## ./camel/retrievers/bm25_retriever.py

Implements a BM25-based retriever class that processes and chunks documents to rank and retrieve relevant text segments based on query term frequency and occurrence.

---

## ./camel/retrievers/cohere_rerank_retriever.py

Implementation of a retriever that uses the Cohere re-ranking model to reorder retrieved documents based on their relevance to a query.

---

## ./camel/retrievers/hybrid_retrival.py

Defines a HybridRetriever class that combines vector-based and BM25 retrieval methods and merges their results using Reciprocal Rank Fusion (RRF) for improved information retrieval.

---

## ./camel/retrievers/vector_retriever.py

Defines a VectorRetriever class that processes and chunks various content types into embeddings stored in a vector database for efficient information retrieval.

---

## ./camel/runtimes/__init__.py

Initialization module that imports and exposes various runtime classes and configurations for the CAMEL AI framework.

---

## ./camel/runtimes/api.py

Defines a FastAPI-based runtime API server that dynamically loads and exposes toolkit functions as asynchronous HTTP endpoints with support for synchronous tool execution in a thread pool.

---

## ./camel/runtimes/base.py

Defines an abstract base class for CAMEL runtimes that manages and interacts with function tools.

---

## ./camel/runtimes/configs.py

Defines a Pydantic data model for configuring task execution parameters inside a container environment.

---

## ./camel/runtimes/daytona_runtime.py

Defines DaytonaRuntime, a class for executing functions within a Daytona sandbox environment using the Daytona API.

---

## ./camel/runtimes/docker_runtime.py

Defines a DockerRuntime class to manage and execute tasks within Docker containers, including mounting directories, copying files, and running commands.

---

## ./camel/runtimes/llm_guard_runtime.py

Defines LLMGuardRuntime, a runtime that uses a language model to evaluate and score the risk level of functions based on their descriptions and parameters for safety assessment.

---

## ./camel/runtimes/remote_http_runtime.py

Defines a RemoteHttpRuntime class that manages and executes functions on a remote HTTP server by starting an API server process and sending function calls via HTTP requests.

---

## ./camel/runtimes/ubuntu_docker_runtime.py

Defines an Ubuntu-specific Docker runtime class that configures environment, Python path, and function execution within Ubuntu-based Docker containers.

---

## ./camel/runtimes/utils/__init__.py

Initialization file for the utils package that imports and exposes FunctionRiskToolkit and IgnoreRiskToolkit classes.

---

## ./camel/runtimes/utils/function_risk_toolkit.py

Defines a toolkit for assessing and reporting the risk levels of functions based on their potential harm.

---

## ./camel/runtimes/utils/ignore_risk_toolkit.py

Defines a toolkit class for managing and selectively ignoring risks associated with specified functions during their next invocation.

---

## ./camel/schemas/__init__.py

Initialization file for the camel.schemas package that imports and exposes OpenAISchemaConverter and OutlinesConverter classes.

---

## ./camel/schemas/base.py

Defines an abstract base class for schema converters that structure input text into specified response formats.

---

## ./camel/schemas/openai_converter.py

Defines OpenAISchemaConverter, a class that uses OpenAI's API to convert text input into structured data conforming to a specified Pydantic BaseModel schema.

---

## ./camel/schemas/outlines_converter.py

Defines the OutlinesConverter class for converting strings or functions into various schema formats using the outlines library across multiple AI model platforms.

---

## ./camel/services/agent_openapi_server.py

Defines a FastAPI server for managing and interacting with ChatAgent instances via OpenAPI endpoints, supporting agent initialization, messaging, and tool integration.

---

## ./camel/societies/__init__.py

This file initializes the societies module by importing and exposing the BabyAGI, RolePlaying, and Workforce classes.

---

## ./camel/societies/babyagi_playing.py

Defines the BabyAGI class implementing a task-driven autonomous agent framework with customizable roles, task management, and agent interactions within the CAMEL AI society.

---

## ./camel/societies/role_playing.py

Defines a RolePlaying class to facilitate role-playing interactions between AI agents with configurable roles, tasks, and optional critic and planner agents.

---

## ./camel/societies/workforce/__init__.py

This file initializes the workforce module by importing and exposing key classes and utilities related to worker roles, task building, workflow management, and failure handling.

---

## ./camel/societies/workforce/base.py

Defines an abstract base class for workforce nodes managing task channels and lifecycle methods in the CAMEL AI framework.

---

## ./camel/societies/workforce/events.py

Defines Pydantic models for various workforce-related event types used to log and track task and worker lifecycle events in the CAMEL AI system.

---

## ./camel/societies/workforce/prompts.py

Defines text prompts for creating worker nodes, assigning tasks with dependencies, and processing tasks within the CAMEL workforce management system.

---

## ./camel/societies/workforce/role_playing_worker.py

Defines a RolePlayingWorker class that processes tasks by simulating a role-playing dialogue between AI assistant and user agents, optionally using structured output handling and summarization.

---

## ./camel/societies/workforce/single_agent_worker.py

Defines a scalable asynchronous pool managing reusable cloned ChatAgent instances with automatic cleanup and concurrency control.

---

## ./camel/societies/workforce/structured_output_handler.py

Defines a handler class for generating prompts and extracting structured JSON output from agent responses using regex patterns and schema validation.

---

## ./camel/societies/workforce/task_channel.py

Defines a TaskChannel class for managing and tracking the status and assignment of tasks wrapped in Packet objects within a workforce system.

---

## ./camel/societies/workforce/utils.py

Utility functions and Pydantic models for managing workforce roles, workflow metadata, configuration, task results, and quality evaluation in the CAMEL AI framework.

---

## ./camel/societies/workforce/worker.py

Defines an abstract asynchronous Worker class for processing tasks within a workforce system, managing task assignment, execution, and state updates.

---

## ./camel/societies/workforce/workflow_memory_manager.py

Manages loading, saving, and intelligent selection of workflow memory for workforce ChatAgent workers, supporting role-based organization and version tracking.

---

## ./camel/societies/workforce/workforce.py

Defines the Workforce class and related components to manage and coordinate multiple worker agents collaboratively solving and decomposing tasks within a task execution system.

---

## ./camel/societies/workforce/workforce_callback.py

Defines an abstract interface for handling and logging various workforce lifecycle events with optional colored output.

---

## ./camel/societies/workforce/workforce_logger.py

Defines the WorkforceLogger class that logs and tracks events, statuses, and metrics related to tasks and workers within a workforce system.

---

## ./camel/societies/workforce/workforce_metrics.py

Defines an abstract base class for workforce metrics with methods to reset data, export to JSON, generate ASCII tree representations, and retrieve key performance indicators.

---

## ./camel/storages/__init__.py

This file initializes the storages package by importing and exposing various key-value, graph, and vector database storage classes and interfaces.

---

## ./camel/storages/graph_storages/__init__.py

This file initializes the graph_storages package by importing and exposing core graph storage classes and elements.

---

## ./camel/storages/graph_storages/base.py

Defines an abstract base class specifying the interface for graph storage systems, including methods for schema management, adding/deleting triplets, and querying.

---

## ./camel/storages/graph_storages/graph_element.py

Defines data models for graph nodes, relationships, and graph elements with associated properties and optional source metadata.

---

## ./camel/storages/graph_storages/nebula_graph.py

Defines a NebulaGraph client class for connecting to, managing sessions with, and executing queries on a NebulaGraph graph database within the CAMEL AI framework.

---

## ./camel/storages/graph_storages/neo4j_graph.py

Defines a Neo4jGraph class for connecting to and interacting with a Neo4j database, including schema retrieval and graph operations, with support for authentication and configuration.

---

## ./camel/storages/key_value_storages/__init__.py

This file initializes the key-value storage module by importing and exposing various storage classes and utilities.

---

## ./camel/storages/key_value_storages/base.py

Defines an abstract base class specifying the interface for key-value storage systems to save, load, and clear data records.

---

## ./camel/storages/key_value_storages/in_memory.py

An in-memory key-value storage implementation for temporary data storage that supports saving, loading, and clearing records.

---

## ./camel/storages/key_value_storages/json.py

Defines a JSON-based key-value storage system with custom serialization for CAMEL-specific enum and Pydantic model types.

---

## ./camel/storages/key_value_storages/mem0_cloud.py

Implementation of a key-value storage backend using Mem0's API to store, search, and manage text-based memories with contextual metadata.

---

## ./camel/storages/key_value_storages/redis.py

Implementation of a Redis-based asynchronous key-value storage class for saving, loading, and clearing records with optional expiration support.

---

## ./camel/storages/object_storages/__init__.py

This file initializes the object_storages package by importing and exposing Amazon S3, Azure Blob, and Google Cloud storage classes.

---

## ./camel/storages/object_storages/amazon_s3.py

Python class for managing AWS S3 object storage, including bucket access, creation, and file operations with support for credentials and anonymous access.

---

## ./camel/storages/object_storages/azure_blob.py

This file defines the AzureBlobStorage class for managing file operations such as upload and download within an Azure Blob Storage container.

---

## ./camel/storages/object_storages/base.py

Abstract base class defining the interface and common methods for object storage operations such as checking existence, uploading, downloading, and file retrieval.

---

## ./camel/storages/object_storages/google_cloud.py

This file defines a Python class for interacting with Google Cloud Storage buckets, including uploading, downloading, and managing files.

---

## ./camel/storages/vectordb_storages/__init__.py

This file initializes the vectordb_storages package by importing and exposing various vector database storage classes and related components.

---

## ./camel/storages/vectordb_storages/base.py

Defines abstract base classes and data models for vector storage operations, including adding, deleting, querying vectors, and tracking vector database status.

---

## ./camel/storages/vectordb_storages/chroma.py

Implementation of a vector storage class for interacting with ChromaDB, supporting multiple client types including in-memory, persistent, HTTP, and cloud for managing vector embeddings.

---

## ./camel/storages/vectordb_storages/faiss.py

Implementation of a FAISS-based vector storage class for efficient similarity search with support for various index types and persistent storage options.

---

## ./camel/storages/vectordb_storages/milvus.py

Implementation of a vector storage class for interacting with Milvus, a cloud-native vector search engine, including client initialization, collection management, and schema setup.

---

## ./camel/storages/vectordb_storages/oceanbase.py

This file defines the OceanBaseStorage class for managing vector data storage and retrieval in an OceanBase vector database, including table creation, indexing, and connection handling.

---

## ./camel/storages/vectordb_storages/pgvector.py

Implementation of a PostgreSQL vector storage using the pgvector extension for adding, querying, and managing vector records with similarity search support.

---

## ./camel/storages/vectordb_storages/qdrant.py

This file implements a QdrantStorage class for managing vector data storage and retrieval using the Qdrant vector search engine, supporting both local and remote clients with configurable options.

---

## ./camel/storages/vectordb_storages/surreal.py

Implementation of a vector storage backend using SurrealDB for scalable, distributed vector storage and similarity search with WebSocket support.

---

## ./camel/storages/vectordb_storages/tidb.py

Implementation of a TiDB-based vector storage class for managing and querying vector data within a TiDB database.

---

## ./camel/storages/vectordb_storages/weaviate.py

Implementation of a vector storage interface for connecting to and interacting with Weaviate vector search engine instances via various connection types.

---

## ./camel/tasks/__init__.py

Initialization module for the camel.tasks package that imports and exposes task-related classes and prompts.

---

## ./camel/tasks/task.py

Defines task validation modes and functions to validate task content and results for quality and failure detection within the CAMEL framework.

---

## ./camel/tasks/task_prompt.py

Defines text prompts for task decomposition, composition, and evolution roles within the CAMEL AI framework.

---

## ./camel/terminators/__init__.py

Initialization module that imports and exposes terminator classes for managing response completion in the CAMEL framework.

---

## ./camel/terminators/base.py

Defines abstract base classes for conversation terminators that determine when a dialogue should end based on specified criteria.

---

## ./camel/terminators/response_terminator.py

Defines a ResponseWordsTerminator class that terminates an agent's response based on specified word occurrence thresholds within the response messages.

---

## ./camel/terminators/token_limit_terminator.py

Defines a terminator class that ends an agent's operation when a specified token usage limit is reached.

---

## ./camel/toolkits/__init__.py

This file initializes the camel.toolkits package by importing and exposing a wide range of toolkit modules and utilities for various functionalities.

---

## ./camel/toolkits/aci_toolkit.py

A Python toolkit class for interacting with the ACI API, providing methods to search for apps and list configured applications with support for authentication and pagination.

---

## ./camel/toolkits/arxiv_toolkit.py

A Python toolkit for searching, retrieving metadata, extracting text, and downloading academic papers from the arXiv API.

---

## ./camel/toolkits/ask_news_toolkit.py

Defines AskNewsToolkit for fetching news and stories via the AskNews API using user queries with support for multiple return formats.

---

## ./camel/toolkits/async_browser_toolkit.py

Defines an asynchronous browser toolkit using Playwright for automated web interactions with support for headless mode, session persistence, and various browser actions.

---

## ./camel/toolkits/audio_analysis_toolkit.py

Defines an AudioAnalysisToolkit class for downloading, transcribing, and analyzing audio files using configurable audio transcription and reasoning models.

---

## ./camel/toolkits/base.py

Defines a base class and utilities for AI toolkits with automatic timeout handling and optional ChatAgent registration support.

---

## ./camel/toolkits/bohrium_toolkit.py

Defines BohriumToolkit, a class for submitting and managing computational jobs on Bohrium services with optional YAML configuration and API key authentication.

---

## ./camel/toolkits/browser_toolkit.py

Defines a BaseBrowser class and utility functions to manage and interact with web browsers using Playwright, supporting features like headless mode, cookie management, and user data persistence.

---

## ./camel/toolkits/browser_toolkit_commons.py

Common utilities, constants, and prompt templates for web browsing agents within the CAMEL toolkit.

---

## ./camel/toolkits/code_execution.py

Defines a toolkit class for executing code in various sandboxed environments with configurable options for safety, verbosity, and execution context.

---

## ./camel/toolkits/context_summarizer_toolkit.py

Defines the ContextSummarizerToolkit class for intelligent conversation context summarization, management, and storage to support agent memory and history handling.

---

## ./camel/toolkits/craw4ai_toolkit.py

Defines an asynchronous toolkit class for web scraping using Crawl4AI's AsyncWebCrawler, providing a scrape method to fetch webpage content and integrating with a function tool interface.

---

## ./camel/toolkits/dappier_toolkit.py

Defines the DappierToolkit class for accessing real-time data and AI-powered recommendations across various domains using the Dappier API.

---

## ./camel/toolkits/data_commons_toolkit.py

Defines a toolkit class for querying and retrieving data from the Data Commons knowledge graph, including SPARQL queries, triples, and statistical time series data.

---

## ./camel/toolkits/dingtalk.py

Python module implementing a toolkit for authenticating with and interacting with the Dingtalk API, including token management, request handling, and webhook signature generation.

---

## ./camel/toolkits/earth_science_toolkit.py

A Python toolkit class providing multiple earth observation science methods, including NDVI and NDWI calculations from satellite raster data.

---

## ./camel/toolkits/edgeone_pages_mcp_toolkit.py

Defines EdgeOnePagesMCPToolkit, a subclass of MCPToolkit, to interface with EdgeOne pages via the EdgeOne Pages MCP server with optional connection timeout.

---

## ./camel/toolkits/excel_toolkit.py

A toolkit class for extracting detailed content and cell information from Excel and CSV files, converting data into Markdown tables, and generating comprehensive reports.

---

## ./camel/toolkits/file_toolkit.py

A cross-platform Python toolkit for comprehensive file operations including reading, writing, editing, and automatic backup of various file formats with customizable encoding and working directory support.

---

## ./camel/toolkits/function_tool.py

Utility functions for generating OpenAI-compatible JSON schemas from Python function signatures and docstrings, supporting asynchronous execution and schema manipulation.

---

## ./camel/toolkits/github_toolkit.py

Defines a toolkit class for interacting with GitHub repositories, including methods to authenticate, retrieve issues, and create pull requests programmatically.

---

## ./camel/toolkits/gmail_toolkit.py

A Python toolkit class for performing various Gmail operations such as sending emails, managing drafts, fetching messages, and handling contacts using the Gmail API.

---

## ./camel/toolkits/google_calendar_toolkit.py

A Python toolkit class for managing Google Calendar events, including creating, retrieving, updating, and deleting events via the Google Calendar API.

---

## ./camel/toolkits/google_drive_mcp_toolkit.py

Defines GoogleDriveMCPToolkit, a subclass of MCPToolkit, to interface with Google Drive via a specified MCP server using optional timeout and credentials.

---

## ./camel/toolkits/google_maps_toolkit.py

Python module defining a GoogleMapsToolkit class for interacting with the Google Maps API, including address validation, elevation retrieval, and timezone information, with built-in exception handling.

---

## ./camel/toolkits/google_scholar_toolkit.py

A Python toolkit class for retrieving and managing author and publication information from Google Scholar using the scholarly module, with support for proxy settings.

---

## ./camel/toolkits/human_toolkit.py

Defines a HumanToolkit class for interactive user communication via console input and message output within the CAMEL framework.

---

## ./camel/toolkits/hybrid_browser_toolkit/__init__.py

Initialization module for the hybrid_browser_toolkit package, exposing the HybridBrowserToolkit class.

---

## ./camel/toolkits/hybrid_browser_toolkit/config_loader.py

Defines dataclasses for browser and toolkit configurations and provides a loader class to create configuration instances from keyword arguments for the HybridBrowserToolkit.

---

## ./camel/toolkits/hybrid_browser_toolkit/hybrid_browser_toolkit.py

Defines a HybridBrowserToolkit class that provides a configurable browser automation interface supporting both TypeScript (WebSocket-based) and Python (Playwright) implementations.

---

## ./camel/toolkits/hybrid_browser_toolkit/hybrid_browser_toolkit_ts.py

Defines a HybridBrowserToolkit class that integrates DOM-based and visual browser automation using a TypeScript Playwright implementation for enhanced AI-driven web interactions.

---

## ./camel/toolkits/hybrid_browser_toolkit/installer.py

Python module for detecting Node.js/npm installation, verifying their availability, and installing necessary npm dependencies for the hybrid browser toolkit.

---

## ./camel/toolkits/hybrid_browser_toolkit/ws_wrapper.py

Asynchronous WebSocket wrapper for managing and logging high-level and low-level browser toolkit actions with memory-aware error handling and process cleanup.

---

## ./camel/toolkits/hybrid_browser_toolkit_py/__init__.py

Initialization file for the hybrid_browser_toolkit_py package that imports and exposes the HybridBrowserToolkit class.

---

## ./camel/toolkits/hybrid_browser_toolkit_py/actions.py

Defines the ActionExecutor class to asynchronously perform and manage various high-level browser actions (e.g., click, type, scroll) on a Playwright Page within a hybrid browser toolkit.

---

## ./camel/toolkits/hybrid_browser_toolkit_py/agent.py

Defines PlaywrightLLMAgent, a web automation assistant that uses a language model to analyze page snapshots, plan actions, and execute browser interactions via a hybrid browser session.

---

## ./camel/toolkits/hybrid_browser_toolkit_py/browser_session.py

Defines a singleton asynchronous wrapper class for managing multi-tab Playwright browser sessions with utilities for snapshots and action execution.

---

## ./camel/toolkits/hybrid_browser_toolkit_py/config_loader.py

Defines a configuration class for browser automation settings, including stealth mode, various timeout durations, action limits, and log limits with support for environment variable overrides.

---

## ./camel/toolkits/hybrid_browser_toolkit_py/hybrid_browser_toolkit.py

Defines the HybridBrowserToolkit class that provides a hybrid browser automation toolkit combining DOM-based and visual screenshot-based interactions for web page control and analysis.

---

## ./camel/toolkits/hybrid_browser_toolkit_py/snapshot.py

Defines a utility class for capturing and diffing YAML-like snapshots of web pages using Playwright in a hybrid browser toolkit.

---

## ./camel/toolkits/image_analysis_toolkit.py

Defines an ImageAnalysisToolkit class that uses vision-capable language models to generate image descriptions and answer questions about images from local paths or URLs.

---

## ./camel/toolkits/image_generation_toolkit.py

A Python toolkit class for generating images using Grok and OpenAI models with customizable parameters and saving options.

---

## ./camel/toolkits/imap_mail_toolkit.py

A toolkit class for managing IMAP email operations including fetching, sending, replying, moving, and deleting emails with connection pooling and idle timeout support.

---

## ./camel/toolkits/jina_reranker_toolkit.py

Defines a toolkit class for reranking documents using the Jina Reranker model via API or local model inference.

---

## ./camel/toolkits/klavis_toolkit.py

Defines KlavisToolkit, a Python class for interacting with the Klavis API to manage MCP server instances, tools, and authentication with HTTP request handling.

---

## ./camel/toolkits/lark_toolkit.py

A Python toolkit class for interacting with Lark (Feishu) chat APIs, handling authentication and API requests.

---

## ./camel/toolkits/linkedin_toolkit.py

A Python toolkit class for interacting with LinkedIn's API to create, delete posts, and retrieve the authenticated user's profile information.

---

## ./camel/toolkits/markitdown_toolkit.py

A deprecated toolkit class for converting various file formats to Markdown, recommending migration to FileToolkit for similar functionality.

---

## ./camel/toolkits/math_toolkit.py

Defines a MathToolkit class providing basic mathematical operations like addition, subtraction, multiplication, division, and rounding, along with deprecated method aliases for backward compatibility.

---

## ./camel/toolkits/mcp_toolkit.py

Defines the MCPToolkit class and related utilities for managing and executing tools via the MCP (Model Control Protocol) client with strict JSON schema validation and error handling.

---

## ./camel/toolkits/memory_toolkit.py

Defines a MemoryToolkit class that manages saving, loading, and clearing a ChatAgent's memory through function tools.

---

## ./camel/toolkits/meshy_toolkit.py

Defines the MeshyToolkit class for interacting with the Meshy API to generate, refine, and monitor 3D model creation tasks from text or images.

---

## ./camel/toolkits/message_agent_toolkit.py

Defines an AgentCommunicationToolkit for managing and facilitating message-based communication between multiple ChatAgent instances in a multi-agent system.

---

## ./camel/toolkits/message_integration.py

Defines a class to integrate user messaging capabilities into CAMEL toolkits and functions, enabling agents to send status updates during toolkit execution with support for custom message handlers.

---

## ./camel/toolkits/microsoft_outlook_mail_toolkit.py

Python module implementing OAuth authentication and token management for Microsoft Outlook Mail integration within the CAMEL AI toolkit.

---

## ./camel/toolkits/mineru_toolkit.py

Defines a toolkit class for extracting and processing document content from URLs and files using the MinerU API, supporting OCR, formula recognition, and table detection.

---

## ./camel/toolkits/minimax_mcp_toolkit.py

Defines MinimaxMCPToolkit, an async Python interface for connecting to MiniMax AI services to access multimedia generation features via the MiniMax MCP server.

---

## ./camel/toolkits/networkx_toolkit.py

This file defines a NetworkXToolkit class that provides an interface for creating and manipulating different types of NetworkX graphs, including adding nodes and edges, retrieving graph elements, and computing shortest paths.

---

## ./camel/toolkits/note_taking_toolkit.py

A Python toolkit for creating, reading, appending, and managing markdown note files stored in a designated directory with a registry system.

---

## ./camel/toolkits/notion_mcp_toolkit.py

Defines NotionMCPToolkit, a subclass of MCPToolkit, to interact asynchronously with Notion via the Model Context Protocol, including schema validation and tool retrieval.

---

## ./camel/toolkits/notion_toolkit.py

A Python toolkit for interacting with the Notion API to retrieve and process user, page, and media information from Notion workspaces.

---

## ./camel/toolkits/open_api_specs/biztoc/__init__.py

Initialization file for the biztoc module within the CAMEL AI open API specifications toolkit.

---

## ./camel/toolkits/open_api_specs/coursera/__init__.py

Initialization module for the Coursera OpenAPI specifications toolkit in the CAMEL project.

---

## ./camel/toolkits/open_api_specs/create_qr_code/__init__.py

Initialization file for the create_qr_code OpenAPI specification toolkit in the CAMEL project.

---

## ./camel/toolkits/open_api_specs/klarna/__init__.py

Initialization module for the Klarna OpenAPI specifications toolkit within the CAMEL project.

---

## ./camel/toolkits/open_api_specs/nasa_apod/__init__.py

Initialization file for the NASA APOD OpenAPI specification toolkit in the CAMEL project.

---

## ./camel/toolkits/open_api_specs/outschool/__init__.py

Initialization file for the Outschool OpenAPI specifications toolkit in the CAMEL project.

---

## ./camel/toolkits/open_api_specs/outschool/paths/__init__.py

Defines API endpoint paths for retrieving classes and searching teachers in the Outschool OpenAPI specification.

---

## ./camel/toolkits/open_api_specs/outschool/paths/get_classes.py

Defines a function to fetch class data from the Outschool API using HTTP GET requests with given parameters.

---

## ./camel/toolkits/open_api_specs/outschool/paths/search_teachers.py

Defines a function to search for teachers on Outschool by making a GET request to the Outschool API with given parameters.

---

## ./camel/toolkits/open_api_specs/security_config.py

Defines security configurations including API key details for OpenAPI specifications used in the CAMEL toolkit.

---

## ./camel/toolkits/open_api_specs/speak/__init__.py

Initialization file for the 'speak' module within the OpenAPI specifications toolkit of the CAMEL project.

---

## ./camel/toolkits/open_api_specs/web_scraper/__init__.py

Initialization module for the web_scraper toolkit within the CAMEL open API specifications.

---

## ./camel/toolkits/open_api_specs/web_scraper/paths/__init__.py

This file initializes the web_scraper paths package within the open_api_specs toolkit of the CAMEL project.

---

## ./camel/toolkits/open_api_specs/web_scraper/paths/scraper.py

Defines a function to send a JSON request to the Scraper API and return the scraped website data or error information.

---

## ./camel/toolkits/open_api_toolkit.py

A Python class that parses OpenAPI specification files and dynamically generates OpenAI-compatible function schemas for interacting with API endpoints defined in the OpenAPI spec.

---

## ./camel/toolkits/openbb_toolkit.py

Python module defining the OpenBBToolkit class for accessing and analyzing financial data via the OpenBB Platform SDK with error handling and search functionalities.

---

## ./camel/toolkits/origene_mcp_toolkit.py

Defines OrigeneToolkit, an async context manager subclass of MCPToolkit for managing connections to Origene MCP servers using a provided configuration.

---

## ./camel/toolkits/playwright_mcp_toolkit.py

Defines PlaywrightMCPToolkit, a subclass of MCPToolkit, to interface with web browsers asynchronously via the Playwright automation library using the Model Context Protocol (MCP).

---

## ./camel/toolkits/pptx_toolkit.py

A Python toolkit class for creating and formatting PowerPoint (PPTX) presentations with support for text styling, image embedding, and file management.

---

## ./camel/toolkits/pubmed_toolkit.py

A Python toolkit for searching and retrieving biomedical papers and metadata from the PubMed MEDLINE database using PubMed's E-utilities API.

---

## ./camel/toolkits/pulse_mcp_search_toolkit.py

A Python toolkit class for searching and retrieving MCP servers via the PulseMCP API with filtering and ranking capabilities.

---

## ./camel/toolkits/pyautogui_toolkit.py

A Python toolkit class for automating GUI interactions using PyAutoGUI with safety boundary checks and mouse control functions.

---

## ./camel/toolkits/reddit_toolkit.py

Defines a RedditToolkit class for interacting with the Reddit API to collect top posts, analyze comment sentiment, and track keyword discussions across subreddits.

---

## ./camel/toolkits/resend_toolkit.py

A Python toolkit for sending emails via the Resend API, supporting HTML and plain text content with multiple recipients and customizable headers.

---

## ./camel/toolkits/retrieval_toolkit.py

Defines a RetrievalToolkit class for retrieving relevant information from local vector storage based on queries using an AutoRetriever.

---

## ./camel/toolkits/screenshot_toolkit.py

A toolkit for capturing, saving, and analyzing screenshots using PIL, integrated with an agent for image interpretation.

---

## ./camel/toolkits/search_toolkit.py

Defines a SearchToolkit class providing web search functionalities using various APIs and services like Serper.dev, Wikipedia, and LinkUp.

---

## ./camel/toolkits/searxng_toolkit.py

A Python toolkit for performing customizable and privacy-respecting web searches using the SearxNG metasearch engine with support for safe search and time range filters.

---

## ./camel/toolkits/semantic_scholar_toolkit.py

A Python toolkit for accessing the Semantic Scholar API to retrieve detailed paper and author information by paper title, paper ID, or bulk queries.

---

## ./camel/toolkits/slack_toolkit.py

Defines a SlackToolkit class for managing Slack operations such as authentication, creating channels, and joining channels using the Slack API.

---

## ./camel/toolkits/sql_toolkit.py

A Python toolkit class for executing SQL queries on DuckDB or SQLite databases with support for read-only mode and configurable connection settings.

---

## ./camel/toolkits/stripe_toolkit.py

Defines a StripeToolkit class for interacting with Stripe API to manage customers, balances, and transactions with built-in logging and error handling.

---

## ./camel/toolkits/sympy_toolkit.py

A toolkit class for performing symbolic mathematics operations such as simplification, expansion, factoring, and solving linear systems using SymPy, with JSON-formatted results and error handling.

---

## ./camel/toolkits/task_planning_toolkit.py

Defines a TaskPlanningToolkit class for decomposing and replanning complex tasks into subtasks within a task management framework.

---

## ./camel/toolkits/terminal_toolkit/__init__.py

Initialization module that imports and exposes the TerminalToolkit class from the terminal_toolkit submodule.

---

## ./camel/toolkits/terminal_toolkit/terminal_toolkit.py

Defines TerminalToolkit, a class enabling LLM agents to execute and interact with terminal commands locally or within Docker containers, supporting sandboxing, session management, and command safety features.

---

## ./camel/toolkits/terminal_toolkit/utils.py

Utility functions for safely checking and sanitizing shell commands, including whitelist enforcement and directory access restrictions.

---

## ./camel/toolkits/thinking_toolkit.py

Defines the ThinkingToolkit class for systematically recording and managing various stages of reasoning such as plans, hypotheses, thoughts, contemplations, critiques, syntheses, and reflections.

---

## ./camel/toolkits/twitter_toolkit.py

Python module providing functions to create and delete tweets via the Twitter API with support for polls and quote tweets, including authentication and error handling.

---

## ./camel/toolkits/vertex_ai_veo_toolkit.py

Python toolkit for generating videos using Google Vertex AI Veo with support for text-to-video and image-to-video generation.

---

## ./camel/toolkits/video_analysis_toolkit.py

Defines a toolkit class for analyzing videos using vision-language models, incorporating frame extraction, audio transcription, OCR, and detailed video content question answering.

---

## ./camel/toolkits/video_download_toolkit.py

Defines a toolkit class for downloading videos using yt-dlp, capturing screenshots from videos, and managing downloaded video files with optional temporary storage.

---

## ./camel/toolkits/weather_toolkit.py

Defines a WeatherToolkit class for fetching and formatting weather data from the OpenWeatherMap API with customizable units and error handling.

---

## ./camel/toolkits/web_deploy_toolkit.py

A Python toolkit for initializing, building, and locally or remotely deploying React web projects with optional branding and server management features.

---

## ./camel/toolkits/wechat_official_toolkit.py

Defines a Python toolkit class for interacting with the WeChat Official Account API, including access token management and API request handling.

---

## ./camel/toolkits/whatsapp_toolkit.py

A Python toolkit class for interacting with the WhatsApp Business API to send messages, retrieve message templates, and access business profile information.

---

## ./camel/toolkits/wolfram_alpha_toolkit.py

A Python toolkit class for querying Wolfram|Alpha API to retrieve simple answers, detailed step-by-step solutions, and LLM-optimized responses with result parsing.

---

## ./camel/toolkits/zapier_toolkit.py

Defines a Python class for interacting with Zapier's NLA API to list, preview, and execute Zapier actions using natural language instructions.

---

## ./camel/types/__init__.py

This file aggregates and exposes various type definitions, enums, and configurations related to models, tasks, and OpenAI chat completions for the CAMEL-AI project.

---

## ./camel/types/agents/__init__.py

Initialization file for the agents module that imports and exposes the ToolCallingRecord class.

---

## ./camel/types/agents/tool_calling_record.py

Defines a Pydantic model for recording and representing historical tool call details, including arguments, results, and optional images.

---

## ./camel/types/enums.py

Defines enumerations for user roles and a comprehensive list of AI model types used in the CAMEL framework.

---

## ./camel/types/mcp_registries.py

Defines configuration models and utilities for different MCP registry types, including Smithery and ACI, with OS-specific command preparation and API key management.

---

## ./camel/types/openai_types.py

This file aggregates and re-exports various OpenAI chat completion-related type definitions and utilities for use within the CAMEL-AI project.

---

## ./camel/types/unified_model_type.py

Defines a thread-safe, cached string subclass to unify and represent various AI model types with properties indicating their platform affiliations and token limits.

---

## ./camel/utils/__init__.py

Initialization module that aggregates and exposes various utility functions, classes, and constants for the CAMEL AI project.

---

## ./camel/utils/agent_context.py

Utility module for managing context-local storage of the current agent ID in both synchronous and asynchronous environments.

---

## ./camel/utils/async_func.py

Utility to convert synchronous Python functions wrapped as FunctionTool instances into asynchronous versions using asyncio.

---

## ./camel/utils/chunker/__init__.py

Initialization module that imports and exposes BaseChunker, CodeChunker, and UnstructuredIOChunker classes for chunking utilities.

---

## ./camel/utils/chunker/base.py

Defines an abstract base class for chunkers with a required chunking method in the CAMEL framework.

---

## ./camel/utils/chunker/code_chunker.py

A Python class that chunks code or text into token-limited segments while preserving structural elements like functions and classes and optionally removing images.

---

## ./camel/utils/chunker/uio_chunker.py

Defines UnstructuredIOChunker, a class for chunking structured text elements into smaller segments while preserving document structure and character limits.

---

## ./camel/utils/commons.py

Utility functions for text processing, task downloading, parsing, and server status checking used in the CAMEL AI project.

---

## ./camel/utils/constants.py

Defines a class containing various constant values used throughout the CAMEL project.

---

## ./camel/utils/context_utils.py

Defines a Pydantic model for structured workflow summaries to capture reusable agent task information and metadata.

---

## ./camel/utils/deduplication.py

Utility functions and data structures for deduplicating text strings based on their embeddings and cosine similarity.

---

## ./camel/utils/filename.py

This file provides a function to sanitize URL paths into safe filenames compatible with most operating systems, handling Unicode normalization, special character replacement, and reserved names.

---

## ./camel/utils/langfuse.py

Utility module for configuring and managing Langfuse tracing integration and agent session context within CAMEL models.

---

## ./camel/utils/mcp.py

Utility module for validating Pydantic serializability of function type annotations and providing a decorator class to register class methods as tools in a Model Context Protocol (MCP) server.

---

## ./camel/utils/mcp_client.py

This file defines a unified MCP client with configurable server connection options supporting multiple transport protocols including stdio, SSE, streamable HTTP, and WebSocket.

---

## ./camel/utils/message_summarizer.py

Defines a utility class and schema for generating structured JSON summaries of chat conversations using a language model backend.

---

## ./camel/utils/response_format.py

Utility functions to generate Pydantic models from JSON schemas, strings, or callable signatures for structured response formatting.

---

## ./camel/utils/token_counting.py

Utility module for counting tokens in messages and text for various AI models, including OpenAI models, with support for encoding and decoding tokens.

---

## ./camel/utils/tool_result.py

Defines a ToolResult class for representing tool outputs that include text and optional base64-encoded images for use in conversational contexts.

---

## ./camel/verifiers/__init__.py

Package initializer that imports and exposes verifier classes and related components for the CAMEL AI project.

---

## ./camel/verifiers/base.py

Defines an abstract asynchronous base class for verifiers with setup, cleanup, retry logic, batch processing, and resource monitoring capabilities.

---

## ./camel/verifiers/math_verifier.py

Defines a MathVerifier class that verifies mathematical expressions, including LaTeX and plain math, with configurable precision and optional LaTeX wrapping using the Math-Verify library.

---

## ./camel/verifiers/models.py

Defines data models and configurations for verification outcomes, results, and verifier behavior using Pydantic and enums.

---

## ./camel/verifiers/physics_verifier.py

Defines a UnitParser class for parsing and handling physical units using SymPy, supporting standard and LaTeX-formatted unit strings with SI prefixes.

---

## ./camel/verifiers/python_verifier.py

Defines a PythonVerifier class that executes and verifies Python code within an isolated virtual environment by setting up the environment, installing dependencies, running the code, and comparing outputs.
