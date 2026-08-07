import { useEffect, useRef, useState } from "react";

// Wrapper scroll-reveal (sama dengan IntersectionObserver di template asli)
export default function Reveal({ children, variant = "", threshold = 0.12, className = "" }) {
  const ref = useRef(null);
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const obs = new IntersectionObserver(
      (entries) => {
        entries.forEach((e) => {
          if (e.isIntersecting) {
            setVisible(true);
            obs.unobserve(e.target);
          }
        });
      },
      { threshold }
    );
    obs.observe(el);
    return () => obs.disconnect();
  }, [threshold]);

  return (
    <div ref={ref} className={`reveal ${variant} ${visible ? "visible" : ""} ${className}`}>
      {children}
    </div>
  );
}
