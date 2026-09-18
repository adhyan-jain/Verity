import { useEffect, useRef, useState } from "react";

export function useTypewriter(
  fullText: string,
  speedMs: number = 16,
  enabled: boolean = true,
  onComplete?: () => void,
) {
  const [displayedText, setDisplayedText] = useState("");
  const [isTyping, setIsTyping] = useState(false);
  const onCompleteRef = useRef(onComplete);
  onCompleteRef.current = onComplete;

  useEffect(() => {
    if (!enabled) {
      setDisplayedText(fullText);
      setIsTyping(false);
      return;
    }

    setDisplayedText("");
    setIsTyping(true);

    let currentIndex = 0;
    const interval = setInterval(() => {
      currentIndex += 1;
      setDisplayedText(fullText.slice(0, currentIndex));

      if (currentIndex >= fullText.length) {
        clearInterval(interval);
        setIsTyping(false);
        if (onCompleteRef.current) {
          onCompleteRef.current();
        }
      }
    }, speedMs);

    return () => clearInterval(interval);
  }, [fullText, speedMs, enabled]);

  return { displayedText, isTyping, isComplete: !isTyping && displayedText.length === fullText.length };
}

export function useSequentialTraceStreamer<T extends { sentence: string }>(
  items: T[],
  speedMs: number = 18,
  enabled: boolean = true,
) {
  const [activeStepIndex, setActiveStepIndex] = useState(0);
  const [streamedMap, setStreamedMap] = useState<Record<number, string>>({});
  const [isDone, setIsDone] = useState(false);

  useEffect(() => {
    if (!enabled || items.length === 0) {
      const all: Record<number, string> = {};
      items.forEach((item, idx) => {
        all[idx] = item.sentence;
      });
      setStreamedMap(all);
      setIsDone(true);
      return;
    }

    setActiveStepIndex(0);
    setStreamedMap({});
    setIsDone(false);

    let step = 0;
    let char = 0;

    const interval = setInterval(() => {
      if (step >= items.length) {
        clearInterval(interval);
        setIsDone(true);
        return;
      }

      const currentItem = items[step];
      if (!currentItem) {
        clearInterval(interval);
        setIsDone(true);
        return;
      }
      const currentText = currentItem.sentence;
      char += 1;

      setStreamedMap((prev) => ({
        ...prev,
        [step]: currentText.slice(0, char),
      }));

      if (char >= currentText.length) {
        step += 1;
        char = 0;
        setActiveStepIndex(step);
      }
    }, speedMs);

    return () => clearInterval(interval);
  }, [items, speedMs, enabled]);

  return { streamedMap, activeStepIndex, isDone };
}
