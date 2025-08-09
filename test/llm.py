import dspy
from app.llm import DspyLLM
from app.models import LLMConfig, Provider


# A tiny signature + module to test generation.
class SimpleQA(dspy.Signature):
    """question -> answer"""

    question: str = dspy.InputField()
    answer: str = dspy.OutputField(desc="usually a short fact")


def main():
    cfg_global = LLMConfig(
        provider=Provider.OLLAMA,
        model="llama3.1",
        extra={"temperature": 0.2},
    )
    DspyLLM(cfg_global).configure_global()

    qa = dspy.Predict(SimpleQA)
    out1 = qa(question="What is 2 + 2?")
    print("[global] answer:", out1.answer)

    #
    cfg_temp = LLMConfig(
        provider=Provider.OLLAMA,
        model="llama3.2",
        extra={"temperature": 0.0},  # make it deterministic(er)
    )
    wrapper = DspyLLM(cfg_temp)
    with wrapper.context():
        qa2 = dspy.Predict(SimpleQA)
        out2 = qa2(question="Name a prime number between 10 and 20.")
        print("[context] answer:", out2.answer)

    out3 = qa(question="Briefly define a hash map.")
    print("[global again] answer:", out3.answer)


if __name__ == "__main__":
    main()
