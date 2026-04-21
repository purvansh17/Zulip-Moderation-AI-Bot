-- Schema migration: align chatsentry DB with data team's new schema
-- Run this on an EXISTING cluster where the DB was created with the old schema.
-- Safe to skip on fresh deployments — postgres.yaml init SQL already has the new schema.
--
-- Run with:
--   kubectl exec -n zulip deploy/postgres -- \
--     psql -U zulip -d chatsentry -f /dev/stdin < migrate_chatsentry_schema.sql
--
-- Or interactively:
--   kubectl exec -it -n zulip deploy/postgres -- psql -U zulip -d chatsentry

BEGIN;

-- 1. Add is_toxicity column (replaces the old multi-label columns)
ALTER TABLE messages ADD COLUMN IF NOT EXISTS is_toxicity BOOLEAN DEFAULT FALSE;

-- 2. Add cleaned_text if missing
ALTER TABLE messages ADD COLUMN IF NOT EXISTS cleaned_text TEXT;

-- 3. Drop old multi-label columns (no longer used)
ALTER TABLE messages DROP COLUMN IF EXISTS toxic;
ALTER TABLE messages DROP COLUMN IF EXISTS severe_toxic;
ALTER TABLE messages DROP COLUMN IF EXISTS obscene;
ALTER TABLE messages DROP COLUMN IF EXISTS threat;
ALTER TABLE messages DROP COLUMN IF EXISTS insult;
ALTER TABLE messages DROP COLUMN IF EXISTS identity_hate;

-- 4. Update source CHECK constraint: 'synthetic_hf' → 'synthetic'
--    Drop the old constraint, add the new one
ALTER TABLE messages DROP CONSTRAINT IF EXISTS messages_source_check;
ALTER TABLE messages ADD CONSTRAINT messages_source_check
    CHECK (source IN ('real', 'synthetic'));

ALTER TABLE users DROP CONSTRAINT IF EXISTS users_source_check;
ALTER TABLE users ADD CONSTRAINT users_source_check
    CHECK (source IN ('real', 'synthetic'));

ALTER TABLE flags DROP CONSTRAINT IF EXISTS flags_source_check;
ALTER TABLE flags ADD CONSTRAINT flags_source_check
    CHECK (source IN ('real', 'synthetic'));

ALTER TABLE moderation DROP CONSTRAINT IF EXISTS moderation_source_check;
ALTER TABLE moderation ADD CONSTRAINT moderation_source_check
    CHECK (source IN ('real', 'synthetic'));

COMMIT;
