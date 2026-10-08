create table conversations (
  id uuid primary key,
  started_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  source text not null,          -- 'deployed' or 'local'
  turns int not null default 0,
  status text not null,          -- in_progress, awaiting_confirm, converted, declined, out_of_scope, no_match, emergency
  category text,
  urgency text,
  providers_matched int,
  avg_turn_seconds real
);
alter table conversations enable row level security;
