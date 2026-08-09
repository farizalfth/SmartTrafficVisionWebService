import { useEffect, useRef, useState } from "react";

export default function CountUp({ end, duration = 1200, decimals = 0, suffix = "", prefix = "", className = "" }) {
  const ref = useRef(null);
  const valRef = useRef(0);
  const [val, setVal] = useState(0);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    let raf;
    const obs = new IntersectionObserver(
      (entries) => {
        entries.forEach((e) => {
          if (!e.isIntersecting) return;
          obs.disconnect();
          const from = valRef.current;
          const t0 = performance.now();
          const tick = (t) => {
            const p = Math.min(1, (t - t0) / duration);
            const eased = 1 - Math.pow(1 - p, 3);
            const next = from + (end - from) * eased;
            valRef.current = next;
            setVal(next);
            if (p < 1) raf = requestAnimationFrame(tick);
          };
          raf = requestAnimationFrame(tick);
        });
      },
      { threshold: 0.3 }
    );
    obs.observe(el);
    return () => {
      obs.disconnect();
      cancelAnimationFrame(raf);
    };
  }, [end, duration]);

  const display = val.toLocaleString("id-ID", {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  });

  return (
    <span ref={ref} className={className}>
      {prefix}
      {display}
      {suffix}
    </span>
  );
}
