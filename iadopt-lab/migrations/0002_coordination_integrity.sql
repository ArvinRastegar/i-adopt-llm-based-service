-- Preserve immutable identities even for the worker role's UPDATE permissions.
SET search_path TO iadopt_lab, public;
CREATE FUNCTION protect_task_identity() RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE actual_attempts integer;
BEGIN
  IF ROW(NEW.id,NEW.campaign_id,NEW.run_id,NEW.provider,NEW.variable_id,NEW.fingerprint)
     IS DISTINCT FROM ROW(OLD.id,OLD.campaign_id,OLD.run_id,OLD.provider,OLD.variable_id,OLD.fingerprint)
  THEN RAISE EXCEPTION 'task scientific identity is immutable' USING ERRCODE='23000'; END IF;
  SELECT count(*) INTO actual_attempts FROM attempt WHERE task_id=NEW.id;
  IF NEW.attempt_count<>actual_attempts OR NEW.attempt_count<OLD.attempt_count
  THEN RAISE EXCEPTION 'attempt counter must equal immutable attempts and never reset' USING ERRCODE='23514'; END IF;
  IF NEW.fence<OLD.fence OR NEW.row_version<OLD.row_version
  THEN RAISE EXCEPTION 'coordination versions cannot decrease' USING ERRCODE='23514'; END IF;
  IF OLD.state='complete' AND NEW.state<>'complete'
  THEN RAISE EXCEPTION 'completed task cannot reopen' USING ERRCODE='23514'; END IF;
  IF NEW.state='complete' AND
     (NOT EXISTS(SELECT 1 FROM prediction WHERE task_id=NEW.id) OR
      NOT EXISTS(SELECT 1 FROM evaluation_item WHERE task_id=NEW.id))
  THEN RAISE EXCEPTION 'task completion requires prediction and evaluation' USING ERRCODE='23514'; END IF;
  RETURN NEW;
END $$;
CREATE TRIGGER task_identity BEFORE UPDATE ON task FOR EACH ROW EXECUTE FUNCTION protect_task_identity();
CREATE TRIGGER task_no_delete BEFORE DELETE ON task FOR EACH ROW EXECUTE FUNCTION reject_evidence_mutation();

CREATE FUNCTION protect_attempt_delivery() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
  IF NEW.attempt_id<>OLD.attempt_id
  THEN RAISE EXCEPTION 'attempt state ownership is immutable' USING ERRCODE='23000'; END IF;
  IF OLD.delivery IN ('response_received','rejected') AND NEW.delivery<>OLD.delivery
  THEN RAISE EXCEPTION 'terminal delivery state cannot reopen' USING ERRCODE='23514'; END IF;
  IF OLD.delivery='ambiguous_delivery' AND NEW.delivery NOT IN ('ambiguous_delivery','response_received','rejected')
  THEN RAISE EXCEPTION 'ambiguous delivery cannot authorize blind redispatch' USING ERRCODE='23514'; END IF;
  IF NEW.delivery='dispatch_started' AND OLD.delivery NOT IN ('not_dispatched','dispatch_started')
  THEN RAISE EXCEPTION 'existing delivery cannot be dispatched again' USING ERRCODE='23514'; END IF;
  RETURN NEW;
END $$;
CREATE TRIGGER attempt_delivery BEFORE UPDATE ON attempt_state FOR EACH ROW EXECUTE FUNCTION protect_attempt_delivery();
CREATE TRIGGER attempt_state_no_delete BEFORE DELETE ON attempt_state FOR EACH ROW EXECUTE FUNCTION reject_evidence_mutation();
ALTER TABLE metric_value ADD CONSTRAINT metric_nonnegative CHECK(numerator>=0 AND value>=0);
