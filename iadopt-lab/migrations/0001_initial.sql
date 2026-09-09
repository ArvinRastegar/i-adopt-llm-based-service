-- PostgreSQL 16. Immutable evidence is separate from mutable coordination rows.
CREATE SCHEMA IF NOT EXISTS iadopt_lab;
SET search_path TO iadopt_lab, public;

CREATE DOMAIN sha256_hex AS text CHECK (VALUE ~ '^[0-9a-f]{64}$');
CREATE TABLE artifact (
  id uuid PRIMARY KEY, kind text NOT NULL, sha256 sha256_hex NOT NULL,
  content bytea NOT NULL, metadata jsonb NOT NULL, evidence_hash sha256_hex NOT NULL,
  created_at timestamptz NOT NULL DEFAULT clock_timestamp(), UNIQUE(kind,sha256)
);
CREATE TABLE corpus_snapshot (
  id uuid PRIMARY KEY, fingerprint sha256_hex UNIQUE NOT NULL,
  repository text NOT NULL, release text NOT NULL, commit_id text NOT NULL,
  tree_id text NOT NULL, expected_count integer NOT NULL CHECK(expected_count > 0),
  evidence jsonb NOT NULL, created_at timestamptz NOT NULL DEFAULT clock_timestamp()
);
CREATE TABLE variable (
  id uuid PRIMARY KEY, corpus_id uuid NOT NULL REFERENCES corpus_snapshot(id),
  variable_id text NOT NULL, source_path text NOT NULL, label text NOT NULL,
  definition text NOT NULL, category text NOT NULL, subcategory text NOT NULL,
  category_path text NOT NULL, source_sha256 sha256_hex NOT NULL,
  gold_sha256 sha256_hex NOT NULL, gold jsonb NOT NULL,
  source_content bytea, demonstration_order integer CHECK(demonstration_order BETWEEN 1 AND 5),
  evidence jsonb NOT NULL, evidence_hash sha256_hex NOT NULL,
  UNIQUE(corpus_id,variable_id), UNIQUE(corpus_id,source_path), UNIQUE(id,corpus_id)
);
CREATE INDEX variable_category_idx ON variable(corpus_id,category,subcategory);

CREATE TABLE campaign (
  id uuid PRIMARY KEY, fingerprint sha256_hex UNIQUE NOT NULL,
  mode text NOT NULL CHECK(mode IN ('synthetic','live')),
  configuration jsonb NOT NULL, configuration_bytes bytea NOT NULL,
  state text NOT NULL DEFAULT 'registered' CHECK(state IN ('registered','planned','running','paused','failed','complete')),
  maximum_cost numeric CHECK(maximum_cost >= 0), currency text NOT NULL,
  spent_cost numeric NOT NULL DEFAULT 0 CHECK(spent_cost >= 0),
  reserved_cost numeric NOT NULL DEFAULT 0 CHECK(reserved_cost >= 0),
  created_at timestamptz NOT NULL DEFAULT clock_timestamp(), updated_at timestamptz NOT NULL DEFAULT clock_timestamp()
);
CREATE TABLE campaign_artifact (
  campaign_id uuid NOT NULL REFERENCES campaign(id), artifact_id uuid NOT NULL REFERENCES artifact(id),
  PRIMARY KEY(campaign_id,artifact_id)
);
CREATE TABLE campaign_provider (
  campaign_id uuid NOT NULL REFERENCES campaign(id), provider text NOT NULL,
  profile jsonb NOT NULL, billing_mode text NOT NULL CHECK(billing_mode IN ('non_billed','metered','synthetic')),
  billing_basis text NOT NULL, maximum_cost numeric CHECK(maximum_cost >= 0),
  currency text NOT NULL, spent_cost numeric NOT NULL DEFAULT 0 CHECK(spent_cost >= 0),
  reserved_cost numeric NOT NULL DEFAULT 0 CHECK(reserved_cost >= 0),
  state text NOT NULL DEFAULT 'ready' CHECK(state IN ('ready','cooldown','paused','failed')),
  reason jsonb, cooldown_until timestamptz,
  PRIMARY KEY(campaign_id,provider), CHECK(provider IN ('psnc','openrouter','mock'))
);
CREATE TABLE model_configuration (
  campaign_id uuid NOT NULL, provider text NOT NULL, model_id text NOT NULL,
  profile jsonb NOT NULL, profile_hash sha256_hex NOT NULL,
  PRIMARY KEY(campaign_id,provider,model_id),
  FOREIGN KEY(campaign_id,provider) REFERENCES campaign_provider(campaign_id,provider)
);
CREATE TABLE resolved_run (
  id uuid PRIMARY KEY, campaign_id uuid NOT NULL REFERENCES campaign(id),
  fingerprint sha256_hex UNIQUE NOT NULL, configuration_id text NOT NULL,
  provider text NOT NULL, model_id text NOT NULL, reasoning_mode text NOT NULL,
  reasoning_fields jsonb NOT NULL, prompt_variant text NOT NULL,
  shot_count integer NOT NULL CHECK(shot_count IN (0,1,3,5)),
  temperature numeric NOT NULL, top_p numeric, max_output_tokens integer,
  repetition integer NOT NULL CHECK(repetition >= 1), evidence jsonb NOT NULL,
  evidence_hash sha256_hex NOT NULL, UNIQUE(id,campaign_id,provider),
  UNIQUE(campaign_id,configuration_id,repetition),
  FOREIGN KEY(campaign_id,provider,model_id) REFERENCES model_configuration(campaign_id,provider,model_id)
);
CREATE TABLE campaign_plan (
  campaign_id uuid PRIMARY KEY REFERENCES campaign(id), fingerprint sha256_hex NOT NULL,
  corpus_id uuid NOT NULL REFERENCES corpus_snapshot(id), population_hash sha256_hex NOT NULL,
  population_size integer NOT NULL CHECK(population_size > 0),
  run_count integer NOT NULL CHECK(run_count > 0), task_count integer NOT NULL CHECK(task_count > 0),
  evidence jsonb NOT NULL, UNIQUE(campaign_id,corpus_id)
);
CREATE TABLE population_member (
  campaign_id uuid NOT NULL REFERENCES campaign_plan(campaign_id),
  variable_id uuid NOT NULL REFERENCES variable(id), position integer NOT NULL CHECK(position >= 0),
  PRIMARY KEY(campaign_id,variable_id), UNIQUE(campaign_id,position)
);
CREATE TABLE worker_session (
  id text PRIMARY KEY, metadata jsonb NOT NULL DEFAULT '{}',
  started_at timestamptz NOT NULL DEFAULT clock_timestamp(), heartbeat_at timestamptz NOT NULL DEFAULT clock_timestamp()
);
CREATE TABLE task (
  id uuid PRIMARY KEY, campaign_id uuid NOT NULL, run_id uuid NOT NULL,
  provider text NOT NULL, variable_id uuid NOT NULL, fingerprint sha256_hex UNIQUE NOT NULL,
  state text NOT NULL DEFAULT 'queued' CHECK(state IN ('queued','request_persisted','generating','response_stored','validated','retry_pending','prediction_ready','complete','operational_failed','ambiguous_delivery','paused_budget','paused_configuration')),
  attempt_count integer NOT NULL DEFAULT 0 CHECK(attempt_count BETWEEN 0 AND 3),
  worker_id text REFERENCES worker_session(id), lease_token uuid, fence bigint NOT NULL DEFAULT 0,
  lease_expires_at timestamptz, heartbeat_at timestamptz, row_version bigint NOT NULL DEFAULT 0,
  reason jsonb, created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
  updated_at timestamptz NOT NULL DEFAULT clock_timestamp(),
  UNIQUE(run_id,variable_id), UNIQUE(id,campaign_id,provider),
  FOREIGN KEY(run_id,campaign_id,provider) REFERENCES resolved_run(id,campaign_id,provider),
  FOREIGN KEY(campaign_id,variable_id) REFERENCES population_member(campaign_id,variable_id),
  CHECK((lease_token IS NULL AND lease_expires_at IS NULL) OR
        (lease_token IS NOT NULL AND lease_expires_at IS NOT NULL AND worker_id IS NOT NULL))
);
CREATE INDEX task_claim_idx ON task(campaign_id,provider,state,lease_expires_at);
CREATE TABLE task_event (
  id uuid PRIMARY KEY, task_id uuid NOT NULL REFERENCES task(id),
  prior_state text, next_state text NOT NULL, cause text NOT NULL, evidence jsonb NOT NULL,
  created_at timestamptz NOT NULL DEFAULT clock_timestamp()
);
CREATE TABLE attempt (
  id uuid PRIMARY KEY, task_id uuid NOT NULL REFERENCES task(id),
  attempt_number integer NOT NULL CHECK(attempt_number BETWEEN 1 AND 3),
  correction_parent uuid, provider text NOT NULL, model_id text NOT NULL,
  messages jsonb NOT NULL, prompt text NOT NULL, request_body jsonb NOT NULL,
  request_evidence jsonb NOT NULL, request_hash sha256_hex NOT NULL,
  scientific_parameters jsonb NOT NULL, scientific_hash sha256_hex NOT NULL,
  idempotency_key text NOT NULL UNIQUE, lease_fence bigint NOT NULL,
  created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
  UNIQUE(task_id,attempt_number), UNIQUE(id,task_id),
  FOREIGN KEY(correction_parent,task_id) REFERENCES attempt(id,task_id)
);
CREATE TABLE attempt_state (
  attempt_id uuid PRIMARY KEY REFERENCES attempt(id),
  delivery text NOT NULL DEFAULT 'not_dispatched' CHECK(delivery IN ('not_dispatched','dispatch_started','response_received','rejected','ambiguous_delivery')),
  dispatched_at timestamptz, received_at timestamptz
);
CREATE TABLE transport_event (
  id uuid PRIMARY KEY, attempt_id uuid NOT NULL REFERENCES attempt(id),
  kind text NOT NULL, evidence jsonb NOT NULL, evidence_hash sha256_hex NOT NULL,
  created_at timestamptz NOT NULL DEFAULT clock_timestamp(), UNIQUE(attempt_id,kind,evidence_hash)
);
CREATE TABLE response (
  id uuid PRIMARY KEY, attempt_id uuid NOT NULL UNIQUE REFERENCES attempt(id),
  raw_body bytea, raw_sha256 sha256_hex, assistant_text text, reasoning_text text,
  envelope jsonb, usage jsonb, http_status integer, returned_model_id text,
  provider_request_id text, finish_reason text, latency_seconds numeric,
  delivery text NOT NULL CHECK(delivery IN ('not_dispatched','response_received','rejected','ambiguous_delivery')),
  evidence jsonb NOT NULL, evidence_hash sha256_hex NOT NULL,
  created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
  CHECK((raw_body IS NULL) = (raw_sha256 IS NULL)),
  CHECK(delivery <> 'response_received' OR raw_body IS NOT NULL)
);
CREATE TABLE validation_event (
  id uuid PRIMARY KEY, attempt_id uuid NOT NULL UNIQUE REFERENCES attempt(id),
  valid boolean NOT NULL, content_invalid boolean NOT NULL,
  candidate jsonb, errors jsonb NOT NULL, evidence jsonb NOT NULL,
  evidence_hash sha256_hex NOT NULL, created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
  CHECK(NOT(valid AND content_invalid))
);
CREATE TABLE prediction (
  id uuid PRIMARY KEY, task_id uuid UNIQUE NOT NULL REFERENCES task(id),
  attempt_id uuid NOT NULL, terminal_invalid boolean NOT NULL,
  canonical jsonb NOT NULL, canonical_hash sha256_hex NOT NULL,
  evidence jsonb NOT NULL, evidence_hash sha256_hex NOT NULL,
  created_at timestamptz NOT NULL DEFAULT clock_timestamp(), UNIQUE(id,task_id),
  FOREIGN KEY(attempt_id,task_id) REFERENCES attempt(id,task_id)
);
CREATE TABLE evaluation_item (
  id uuid PRIMARY KEY, task_id uuid NOT NULL REFERENCES task(id),
  prediction_id uuid NOT NULL, scorer_version text NOT NULL,
  evidence jsonb NOT NULL, evidence_hash sha256_hex NOT NULL,
  created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
  UNIQUE(task_id,scorer_version), FOREIGN KEY(prediction_id,task_id) REFERENCES prediction(id,task_id)
);
CREATE TABLE metric_value (
  evaluation_id uuid NOT NULL REFERENCES evaluation_item(id), mode text NOT NULL CHECK(mode IN ('exact','close')),
  component text NOT NULL, metric text NOT NULL CHECK(metric IN ('tp','fp','fn','tn','precision','recall','f1')),
  numerator numeric NOT NULL CHECK(numerator=trunc(numerator)),
  denominator numeric NOT NULL CHECK(denominator > 0 AND denominator=trunc(denominator)),
  value numeric NOT NULL, PRIMARY KEY(evaluation_id,mode,component,metric)
);
CREATE TABLE live_authorization (
  id uuid PRIMARY KEY, campaign_id uuid NOT NULL REFERENCES campaign(id),
  plan_fingerprint sha256_hex NOT NULL, estimate jsonb NOT NULL, estimate_hash sha256_hex NOT NULL,
  disclosure jsonb NOT NULL, authorization_receipt jsonb NOT NULL, evidence_hash sha256_hex NOT NULL UNIQUE,
  created_at timestamptz NOT NULL DEFAULT clock_timestamp()
);
CREATE TABLE cost_reservation (
  attempt_id uuid PRIMARY KEY REFERENCES attempt(id), campaign_id uuid NOT NULL,
  provider text NOT NULL, amount numeric NOT NULL CHECK(amount>=0),
  currency text NOT NULL, bounded boolean NOT NULL, basis jsonb NOT NULL,
  authorization_id uuid REFERENCES live_authorization(id),
  FOREIGN KEY(campaign_id,provider) REFERENCES campaign_provider(campaign_id,provider)
);
CREATE TABLE cost_settlement (
  attempt_id uuid PRIMARY KEY REFERENCES cost_reservation(attempt_id),
  amount numeric CHECK(amount>=0), state text NOT NULL CHECK(state IN ('confirmed_zero','actual','estimated','unavailable','ambiguous')),
  evidence jsonb NOT NULL, evidence_hash sha256_hex NOT NULL,
  created_at timestamptz NOT NULL DEFAULT clock_timestamp()
);
CREATE TABLE provider_event (
  id uuid PRIMARY KEY, campaign_id uuid NOT NULL, provider text NOT NULL,
  state text NOT NULL, evidence jsonb NOT NULL, created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
  FOREIGN KEY(campaign_id,provider) REFERENCES campaign_provider(campaign_id,provider)
);
CREATE TABLE ranking_run (
  id uuid PRIMARY KEY, campaign_id uuid NOT NULL REFERENCES campaign(id),
  evidence jsonb NOT NULL, evidence_hash sha256_hex UNIQUE NOT NULL,
  created_at timestamptz NOT NULL DEFAULT clock_timestamp()
);
CREATE TABLE configuration_rank (
  ranking_id uuid NOT NULL REFERENCES ranking_run(id), configuration_id text NOT NULL,
  provider text NOT NULL, model_id text NOT NULL, rank integer CHECK(rank>=1),
  reason text, primary_numerator numeric, primary_denominator numeric,
  evidence jsonb NOT NULL, PRIMARY KEY(ranking_id,configuration_id),
  CHECK((primary_numerator IS NULL AND primary_denominator IS NULL) OR
        (primary_numerator=trunc(primary_numerator) AND primary_denominator>0 AND primary_denominator=trunc(primary_denominator)))
);
CREATE TABLE report_manifest (
  id uuid PRIMARY KEY, campaign_id uuid NOT NULL REFERENCES campaign(id),
  evidence jsonb NOT NULL, evidence_hash sha256_hex UNIQUE NOT NULL,
  final boolean NOT NULL, created_at timestamptz NOT NULL DEFAULT clock_timestamp()
);

CREATE FUNCTION reject_evidence_mutation() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN RAISE EXCEPTION 'immutable experiment evidence cannot be %', TG_OP USING ERRCODE='23000'; END $$;
DO $$ DECLARE t text; BEGIN
  FOREACH t IN ARRAY ARRAY['artifact','corpus_snapshot','variable','campaign_artifact','model_configuration',
    'resolved_run','campaign_plan','population_member','task_event','attempt','transport_event',
    'response','validation_event','prediction','evaluation_item','metric_value','live_authorization',
    'cost_reservation','cost_settlement','provider_event','ranking_run','configuration_rank','report_manifest'] LOOP
    EXECUTE format('CREATE TRIGGER immutable_evidence BEFORE UPDATE OR DELETE ON %I FOR EACH ROW EXECUTE FUNCTION reject_evidence_mutation()',t);
  END LOOP;
END $$;
CREATE FUNCTION protect_campaign_identity() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
  IF ROW(NEW.id,NEW.fingerprint,NEW.mode,NEW.configuration,NEW.configuration_bytes,NEW.maximum_cost,NEW.currency)
     IS DISTINCT FROM ROW(OLD.id,OLD.fingerprint,OLD.mode,OLD.configuration,OLD.configuration_bytes,OLD.maximum_cost,OLD.currency)
  THEN RAISE EXCEPTION 'campaign identity is immutable' USING ERRCODE='23000'; END IF;
  RETURN NEW;
END $$;
CREATE TRIGGER campaign_identity BEFORE UPDATE ON campaign FOR EACH ROW EXECUTE FUNCTION protect_campaign_identity();
CREATE TRIGGER campaign_no_delete BEFORE DELETE ON campaign FOR EACH ROW EXECUTE FUNCTION reject_evidence_mutation();
CREATE FUNCTION protect_provider_identity() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
  IF ROW(NEW.campaign_id,NEW.provider,NEW.profile,NEW.billing_mode,NEW.billing_basis,NEW.maximum_cost,NEW.currency)
     IS DISTINCT FROM ROW(OLD.campaign_id,OLD.provider,OLD.profile,OLD.billing_mode,OLD.billing_basis,OLD.maximum_cost,OLD.currency)
  THEN RAISE EXCEPTION 'provider identity is immutable' USING ERRCODE='23000'; END IF;
  RETURN NEW;
END $$;
CREATE TRIGGER provider_identity BEFORE UPDATE ON campaign_provider FOR EACH ROW EXECUTE FUNCTION protect_provider_identity();
CREATE TRIGGER provider_no_delete BEFORE DELETE ON campaign_provider FOR EACH ROW EXECUTE FUNCTION reject_evidence_mutation();
CREATE FUNCTION check_attempt_lineage() RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE owned_provider text; owned_model text; prior attempt%ROWTYPE;
BEGIN
  SELECT r.provider,r.model_id INTO owned_provider,owned_model FROM task t JOIN resolved_run r ON r.id=t.run_id WHERE t.id=NEW.task_id FOR UPDATE OF t;
  IF owned_provider IS DISTINCT FROM NEW.provider OR owned_model IS DISTINCT FROM NEW.model_id
  THEN RAISE EXCEPTION 'attempt provider/model ownership mismatch' USING ERRCODE='23503'; END IF;
  IF NEW.attempt_number=1 AND NEW.correction_parent IS NOT NULL
  THEN RAISE EXCEPTION 'first attempt cannot have correction parent' USING ERRCODE='23514'; END IF;
  IF NEW.attempt_number>1 THEN
    SELECT * INTO prior FROM attempt WHERE task_id=NEW.task_id AND attempt_number=NEW.attempt_number-1;
    IF NOT FOUND OR prior.scientific_hash<>NEW.scientific_hash
    THEN RAISE EXCEPTION 'nonconsecutive attempt or scientific drift' USING ERRCODE='23514'; END IF;
    IF NEW.correction_parent IS NOT NULL AND NEW.correction_parent<>prior.id
    THEN RAISE EXCEPTION 'correction parent must immediately precede attempt' USING ERRCODE='23514'; END IF;
  END IF;
  RETURN NEW;
END $$;
CREATE TRIGGER attempt_lineage BEFORE INSERT ON attempt FOR EACH ROW EXECUTE FUNCTION check_attempt_lineage();

CREATE VIEW source_file AS SELECT id,corpus_id,source_path,source_sha256,source_content,category_path FROM variable;
CREATE VIEW gold_decomposition AS SELECT id AS variable_id,corpus_id,gold,gold_sha256 FROM variable;
CREATE VIEW science_category AS SELECT DISTINCT corpus_id,category,subcategory,category_path FROM variable;
CREATE VIEW evaluation_facts AS
SELECT e.id AS evaluation_id,t.campaign_id,r.configuration_id,r.provider,r.model_id,
       r.reasoning_mode,r.prompt_variant,r.shot_count,r.temperature,r.repetition,
       v.variable_id,v.category,v.subcategory,v.category_path,v.source_path,
       m.mode,m.component,m.metric,m.numerator,m.denominator,m.value
FROM evaluation_item e JOIN task t ON t.id=e.task_id JOIN resolved_run r ON r.id=t.run_id
JOIN variable v ON v.id=t.variable_id JOIN metric_value m ON m.evaluation_id=e.id;
