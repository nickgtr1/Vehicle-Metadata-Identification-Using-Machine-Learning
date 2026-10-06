import argparse

from google import genai
from google.genai import types

# gemini-3.1-flash-lite: only region confirmed as of writing to support supervised
# tuning for this model. If Google adds more regions later, pass --location to override.
DEFAULT_LOCATION = "us-central1"
BASE_MODEL = "gemini-3.1-flash-lite"


def make_client(project: str, location: str) -> genai.Client:
    return genai.Client(vertexai=True, project=project, location=location)


def cmd_start(args):
    client = make_client(args.project, args.location)
    adapter_size = f"ADAPTER_SIZE_{args.adapter_size}" if args.adapter_size else None
    config = types.CreateTuningJobConfig(
        tuned_model_display_name=args.display_name,
        validation_dataset=types.TuningValidationDataset(gcs_uri=args.val_uri) if args.val_uri else None,
        epoch_count=args.epochs,  # omit / leave None to let Vertex pick a recommended value
        adapter_size=adapter_size,  # omit / leave None to let Vertex pick a recommended value
    )
    job = client.tunings.tune(
        base_model=BASE_MODEL,
        training_dataset=types.TuningDataset(gcs_uri=args.train_uri),
        config=config,
    )
    print(f"Tuning job created.\n  name:  {job.name}\n  state: {job.state}")
    print("\nThis job now runs on Google's servers. Closing this terminal is safe.")
    print("Check progress with:")
    print(f'  python scripts/run_tuning_job.py status --project {args.project} --job-name "{job.name}"')


def cmd_status(args):
    client = make_client(args.project, args.location)
    job = client.tunings.get(name=args.job_name)
    print(f"name:  {job.name}")
    print(f"state: {job.state}")
    if getattr(job, "error", None):
        print(f"error: {job.error}")
    tuned_model = getattr(job, "tuned_model", None)
    if tuned_model is not None:
        endpoint = getattr(tuned_model, "endpoint", None)
        model_name = getattr(tuned_model, "model", None)
        if endpoint or model_name:
            print("\nTuned model is ready. Save these for client.py:")
            if model_name:
                print(f"  model:    {model_name}")
            if endpoint:
                print(f"  endpoint: {endpoint}")


def cmd_list(args):
    client = make_client(args.project, args.location)
    found = False
    for job in client.tunings.list():
        found = True
        print(f"{job.name}  [{job.state}]  {getattr(job, 'tuned_model_display_name', '')}")
    if not found:
        print("No tuning jobs found on this project/location.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", required=True)
    ap.add_argument("--location", default=DEFAULT_LOCATION)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_start = sub.add_parser("start")
    p_start.add_argument("--train-uri", required=True)
    p_start.add_argument("--val-uri", default=None)
    p_start.add_argument("--display-name", default="vehicle-id-flash-lite")
    p_start.add_argument("--epochs", type=int, default=None, help="omit to let Vertex pick a recommended value")
    p_start.add_argument(
        "--adapter-size", default=None,
        choices=["ONE", "TWO", "FOUR", "EIGHT", "SIXTEEN", "THIRTY_TWO"],
        help="LoRA adapter rank (bigger = more capacity to change behaviour, also more cost/overfit risk). Vertex auto-picked TWO on the first run; omit to let it choose again.",
    )
    p_start.set_defaults(func=cmd_start)

    p_status = sub.add_parser("status")
    p_status.add_argument("--job-name", required=True)
    p_status.set_defaults(func=cmd_status)

    p_list = sub.add_parser("list")
    p_list.set_defaults(func=cmd_list)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()