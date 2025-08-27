-- WARNING: This schema is for context only and is not meant to be run.
-- Table order and constraints may not be valid for execution.

CREATE TABLE public.causal_insights (
  id uuid NOT NULL DEFAULT gen_random_uuid(),
  hypothesis_id uuid NOT NULL,
  analysis_timestamp timestamp with time zone DEFAULT now(),
  causal_factor_identified text NOT NULL,
  causal_strength double precision,
  recommendation_for_future_ideation text,
  CONSTRAINT causal_insights_pkey PRIMARY KEY (id),
  CONSTRAINT causal_insights_hypothesis_id_fkey FOREIGN KEY (hypothesis_id) REFERENCES public.hypotheses(id)
);
CREATE TABLE public.data_sources (
  id uuid NOT NULL DEFAULT gen_random_uuid(),
  ingestion_timestamp timestamp with time zone DEFAULT now(),
  type text NOT NULL,
  source_url text,
  raw_content_path text,
  processed_insights_path text,
  embedding USER-DEFINED,
  CONSTRAINT data_sources_pkey PRIMARY KEY (id)
);
CREATE TABLE public.human_feedback (
  id uuid NOT NULL DEFAULT gen_random_uuid(),
  hypothesis_id uuid NOT NULL,
  associated_validation_result_id uuid,
  feedback_timestamp timestamp with time zone DEFAULT now(),
  human_decision text NOT NULL,
  rationale_text text,
  CONSTRAINT human_feedback_pkey PRIMARY KEY (id),
  CONSTRAINT human_feedback_associated_validation_result_id_fkey FOREIGN KEY (associated_validation_result_id) REFERENCES public.validation_results(id),
  CONSTRAINT human_feedback_hypothesis_id_fkey FOREIGN KEY (hypothesis_id) REFERENCES public.hypotheses(id)
);
CREATE TABLE public.hypotheses (
  id uuid NOT NULL DEFAULT gen_random_uuid(),
  ideation_timestamp timestamp with time zone DEFAULT now(),
  generated_by_agent text,
  initial_hypothesis_text text NOT NULL,
  current_status text DEFAULT 'pending_validation'::text,
  validation_tier_progress integer DEFAULT 0,
  causal_analysis_summary text,
  human_feedback_id uuid,
  CONSTRAINT hypotheses_pkey PRIMARY KEY (id),
  CONSTRAINT hypotheses_human_feedback_id_fkey FOREIGN KEY (human_feedback_id) REFERENCES public.human_feedback(id)
);
CREATE TABLE public.mlops_metadata (
  id uuid NOT NULL DEFAULT gen_random_uuid(),
  experiment_id text,
  run_id text,
  model_version text,
  metrics_json jsonb,
  parameters_json jsonb,
  artifact_paths text,
  CONSTRAINT mlops_metadata_pkey PRIMARY KEY (id)
);
CREATE TABLE public.validation_results (
  id uuid NOT NULL DEFAULT gen_random_uuid(),
  hypothesis_id uuid NOT NULL,
  tier integer NOT NULL,
  validation_timestamp timestamp with time zone DEFAULT now(),
  metrics_json jsonb,
  pass_fail_status text,
  human_override_flag boolean DEFAULT false,
  CONSTRAINT validation_results_pkey PRIMARY KEY (id),
  CONSTRAINT validation_results_hypothesis_id_fkey FOREIGN KEY (hypothesis_id) REFERENCES public.hypotheses(id)
);
