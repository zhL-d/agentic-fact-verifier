import { createRoot } from "react-dom/client";
import { App } from "./App";
import "./style.css";

const page = document.getElementById("page");
if (!page) throw new Error("Missing #page root element");

createRoot(page).render(<App />);
