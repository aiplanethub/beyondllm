from dataclasses import dataclass
try:
    import phoenix as px
    from phoenix.trace.openai import OpenAIInstrumentor
except ImportError:
    user_agree = input("The feature you're trying to use requires additional packages. Would you like to install them now? [y/N]: ")
    if user_agree.lower() == 'y':
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "arize-phoenix[evals]"])

        import phoenix as px
        from phoenix.trace.openai import OpenAIInstrumentor
    else:
        raise ImportError("The required Phoenix packages are not installed.")
    


@dataclass
class Observer:
    instrumentor: OpenAIInstrumentor = OpenAIInstrumentor()

    def run(self):
        self.instrumentor.instrument()
        self.session = px.launch_app()
        