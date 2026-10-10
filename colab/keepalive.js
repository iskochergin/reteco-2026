// Clicker for Colab: keeps the session from disconnecting when idle while the runtime is connected.
// Paste it into the console of the Colab tab (Cmd+Option+J → Console tab → paste → Enter)
// after the notebook has been started. Once a minute it clicks the resources button in the header.
// It stops by itself once the runtime has disconnected (the notebook finished or crashed and
// released the GPU) and does NOT reconnect, so no compute units are spent on an idle machine.
(() => {
  if (window.__retecoKeepAlive) { console.log("clicker is already running"); return; }
  const btn = () => document.querySelector("colab-connect-button")?.shadowRoot?.querySelector("#connect");
  const disconnected = () => {
    const b = btn();
    if (!b) return true;
    // the button also shows the GPU type ("Reconnect T4"), so only the first word is compared;
    // the non-English labels below are the same button texts in the Russian-locale Colab UI
    const text = (b.textContent || "").trim().split(/\s+/)[0].toLowerCase();
    const tip = (b.getAttribute("tooltiptext") || "").toLowerCase();
    return ["connect", "reconnect", "подключиться", "переподключиться"].includes(text)
      || tip.startsWith("connect to a new") || tip.startsWith("подключиться к");
  };
  let seen = false, clicks = 0;
  window.__retecoKeepAlive = setInterval(() => {
    if (!disconnected()) {
      seen = true;
      btn().click();                       // opens/closes the resources panel; harmless
      clicks += 1;
      console.log(`keep-alive ${new Date().toLocaleTimeString()} · ${clicks}`);
    } else if (seen) {
      clearInterval(window.__retecoKeepAlive);
      window.__retecoKeepAlive = null;
      console.log("runtime disconnected: clicker stopped");
    }
  }, 60 * 1000);
  console.log("clicker started: once a minute; stops by itself after the runtime disconnects");
})();
