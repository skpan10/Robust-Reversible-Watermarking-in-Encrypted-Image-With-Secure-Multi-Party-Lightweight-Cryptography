import React from "react";
import { Layout, Icon } from "antd";
import FooterCSS from "./Footer.module.css";

const Footer = () => {
	return (
		<Layout.Footer className={FooterCSS.footer}>
			<div className={FooterCSS.footerContainer}>
				<p>
					Made for Project Exhibition by{" "}
					<a href="https://github.com/Abhinavv16">Team 30</a>
				</p>
				<a href="https://github.com/Abhinavv16">Thank You </a>
			</div>
		</Layout.Footer>
	);
};

export default Footer;
